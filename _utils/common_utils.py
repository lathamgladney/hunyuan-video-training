"""
Module for handling LoRA model operations
"""

from dataclasses import dataclass
import json
import os
from queue import Queue
import re
import glob
import threading
import modal
import requests
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from modal.volume import FileEntryType
import toml
from _utils.model_utils import parse_hf_url
from _utils.constants import (
    ModelTypes, ModelPaths, Volumes, 
    TRAINING_PATHS
    )
import gradio as gr
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

@dataclass
class StepInfo:
    """Class containing step information for a folder"""
    has_steps: bool
    latest_step: int
    folder_name: str
    dataset_name: str = ""

@dataclass
class EpochInfo:
    """Class containing epoch information for a folder"""
    has_epochs: bool
    latest_epoch: int
    epoch_list: List[int]
    total_epochs: int
    folder_name: str
    dataset_name: str = ""

# Global caches for optimization
step_cache = {}
step_cache_lock = threading.Lock()

epoch_cache = {}
epoch_cache_lock = threading.Lock()

test_outputs_cache = {}
test_outputs_cache_lock = threading.Lock()

# Add cache for training folders
training_folders_cache = {}
training_folders_cache_lock = threading.Lock()

# Add cache for folders with epochs
epochs_folders_cache = {}
epochs_folders_cache_lock = threading.Lock()

# Constants for step processing
MAX_WORKERS = 8  # Maximum number of threads
BATCH_SIZE = 10  # Number of folders to process per batch
MAX_RETRIES = 3  # Maximum retry attempts for volume operations
RETRY_DELAY = 1  # Delay between retries in seconds
FOLDER_PATTERN = r'^\d{8}_\d{2}-\d{2}-\d{2}'  # Pattern for training folders
STEP_PATTERN = r'^global_step\d+'  # Pattern for step folders
EPOCH_PATTERN = r'^epoch\d+'  # Pattern for epoch folders

class FolderProcessor:
    """Class for processing and analyzing training folders"""
    
    def __init__(self, volume):
        self.volume = volume
        self.logger = logging.getLogger(__name__)
        self._connection_pool = Queue(maxsize=MAX_WORKERS)
        self._init_connection_pool()
        
    def _init_connection_pool(self):
        """Initialize connection pool"""
        for _ in range(MAX_WORKERS):
            self._connection_pool.put(modal.Volume.from_name(Volumes.TRAINING))
            
    def _get_connection(self):
        """Get connection from pool"""
        return self._connection_pool.get()
        
    def _return_connection(self, conn):
        """Return connection to pool"""
        self._connection_pool.put(conn)

    def _retry_operation(self, operation, *args, **kwargs):
        """Retry mechanism for volume operations"""
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        raise last_exception

    def _is_potential_training_folder(self, folder_name: str) -> bool:
        """Quick check if folder could be a training folder"""
        return bool(re.match(FOLDER_PATTERN, folder_name))

    def _process_single_folder_steps(self, folder_name: str, force_reload: bool = False) -> Optional[StepInfo]:
        """Process a single folder for steps with caching"""
        with step_cache_lock:
            if folder_name in step_cache and not force_reload:
                return step_cache[folder_name]
        
        try:
            # Fast fail if folder doesn't match format
            if not self._is_potential_training_folder(folder_name):
                return None
                
            # Get latest step from file
            latest_step = get_latest_step_from_file(folder_name)
            if latest_step is None:
                return None
                
            # Get dataset name
            dataset_name = get_dataset_name(folder_name)
            
            step_info = StepInfo(
                has_steps=True,
                latest_step=latest_step,
                folder_name=folder_name,
                dataset_name=dataset_name
            )
            
            with step_cache_lock:
                step_cache[folder_name] = step_info
            
            return step_info
            
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_name} for steps: {str(e)}")
            return None
    def _process_single_folder_epochs(self, folder_name: str, force_reload: bool = False) -> Optional[EpochInfo]:
        """Process a single folder for epochs with caching"""
        with epoch_cache_lock:
            if folder_name in epoch_cache and not force_reload:
                return epoch_cache[folder_name]
        
        try:
            # Fast fail if folder doesn't match format
            if not self._is_potential_training_folder(folder_name):
                return None
                
            conn = self._get_connection()
            try:
                # List folder contents with retry
                entries = self._retry_operation(
                    conn.listdir,
                    folder_name,
                    recursive=False
                )
                
                # Find epoch folders with pattern matching
                epoch_numbers = []
                for entry in entries:
                    if entry.type != FileEntryType.FILE:
                        folder_base = os.path.basename(entry.path.rstrip("/"))
                        if re.match(EPOCH_PATTERN, folder_base):
                            try:
                                epoch_num = int(folder_base[5:])
                                epoch_numbers.append(epoch_num)
                            except ValueError:
                                continue
                
                if not epoch_numbers:
                    return None
                    
                # Get dataset name
                dataset_name = get_dataset_name(folder_name)
                
                # Sort epoch numbers
                epoch_numbers.sort()
                
                epoch_info = EpochInfo(
                    has_epochs=True,
                    latest_epoch=max(epoch_numbers),
                    epoch_list=epoch_numbers,
                    total_epochs=len(epoch_numbers),
                    folder_name=folder_name,
                    dataset_name=dataset_name
                )
                
                with epoch_cache_lock:
                    epoch_cache[folder_name] = epoch_info
                
                return epoch_info
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_name} for epochs: {str(e)}")
            return None
    def process_folders_parallel(self, folders: List[str], process_type: str = "steps", force_reload: bool = False) -> List[Union[StepInfo, EpochInfo]]:
        """Process multiple folders in parallel using batches
        
        Args:
            folders: List of folder names to process
            process_type: Type of processing ("steps" or "epochs")
            force_reload: If True, bypass cache and reload from volume
        """
        results = []
        process_func = (
            self._process_single_folder_steps if process_type == "steps"
            else self._process_single_folder_epochs
        )
        
        # Split folders into batches
        for i in range(0, len(folders), BATCH_SIZE):
            batch = folders[i:i + BATCH_SIZE]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit batch of folders for processing
                future_to_folder = {
                    executor.submit(process_func, folder, force_reload): folder 
                    for folder in batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_folder):
                    folder = future_to_folder[future]
                    try:
                        info = future.result()
                        if info:
                            results.append(info)
                    except Exception as e:
                        self.logger.error(f"Error processing folder {folder}: {str(e)}")
                        
        return results

def upload_config_files(config_volume, config_dir="config"):
    workflow_file = Path(config_dir) / "workflow_api.json"
    if not workflow_file.exists():
        raise FileNotFoundError(f"Required workflow file {workflow_file} not found")

    with config_volume.batch_upload(force=True) as batch:
        # Upload workflow file directly to volume root
        batch.put_file(str(workflow_file), "workflow_api.json")
        
        # Upload toml files directly to volume root
        for config_file in Path(config_dir).glob("*.toml"):
            batch.put_file(str(config_file), config_file.name)
    logger.info(f"Uploaded config files and workflow to {config_volume}")

def upload_dataset(data_volume, dataset_dir):
    """
    Upload all files in the dataset directory to the Modal data_volume.
    
    Args:
        data_volume: Modal volume for uploading files.
        dataset_dir (str): Path to the directory containing the dataset.
    """
    with data_volume.batch_upload(force=True) as batch:
        for data_file in Path(dataset_dir).rglob("*"):
            if data_file.is_file():
                remote_path = f"/{data_file.relative_to(dataset_dir)}"
                batch.put_file(str(data_file), remote_path)
    logger.info(f"Uploaded dataset to {data_volume} volume")
    
# Add common files to the list of files to upload to HuggingFace
def add_common_files(training_folder: str, paths_to_upload: list) -> None:
    common_files = ["hunyuan_video.toml", "latest", "test_outputs"]
    for file in common_files:
        file_path = os.path.join(training_folder, file)
        if os.path.exists(file_path) and file_path not in paths_to_upload:
            paths_to_upload.append(file_path)

# Upload files to HuggingFace
def upload_to_huggingface(path_or_fileobj: str, path_in_repo: str, repo_id: str, token: str, private_repo: bool = True):
    api = HfApi()
    try:
        create_repo(repo_id=repo_id, private=private_repo, token=token)
    except Exception:
        pass
    api.upload_file(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo, repo_id=repo_id, token=token)

def find_latest_training_folder(output_dir: str) -> str:
    """Find the latest training folder based on timestamp in folder name
    
    Args:
        output_dir: Base output directory
        
    Returns:
        str: Path to latest training folder
    """
    # Get all folders with "202*"
    folders = glob.glob(os.path.join(output_dir, "202*"))
    if not folders:
        return ""
        
    # Sort by folder name (timestamp) in descending order
    latest_folder = sorted(folders, reverse=True)[0]
    logger.info(f"Latest training folder: {os.path.basename(latest_folder)}")
    return latest_folder

def find_lora_checkpoints(output_dir: str) -> List[Tuple[str, int]]:
    """Find all LoRA checkpoints in the output directory
    
    Args:
        output_dir: Directory containing training outputs or specific training folder
        
    Returns:
        List[Tuple[str, int]]: List of (checkpoint_path, epoch_number) tuples
    """
    checkpoints = []
    
    # Check if output_dir is already a training folder
    if os.path.basename(output_dir).startswith("202"):  # Format: YYYYMMDD_HH-MM-SS
        training_folder = output_dir
    else:
        # Find latest training folder
        training_dirs = sorted(glob.glob(os.path.join(output_dir, "202*")))
        if not training_dirs:
            logger.warning("No training folders found")
            return checkpoints
        training_folder = training_dirs[-1]
        
    logger.info(f"Searching for checkpoints in: {training_folder}")
    
    # Find all epoch folders
    epoch_pattern = re.compile(r"epoch(\d+)")
    for folder in glob.glob(os.path.join(training_folder, "epoch*")):
        match = epoch_pattern.search(folder)
        if match:
            epoch_num = int(match.group(1))
            adapter_path = os.path.join(folder, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                logger.info(f"Found epoch {epoch_num}")
                checkpoints.append((adapter_path, epoch_num))
                
    logger.info(f"Total checkpoints found in {os.path.basename(training_folder)}: {len(checkpoints)}")
    
    # Sort by epoch number
    return sorted(checkpoints, key=lambda x: x[1])

def get_epochs_to_test(checkpoints: List[Tuple[str, int]], epoch_config: List[int]) -> List[Tuple[str, int]]:
    """Get list of epochs to test based on configuration"""
    if not epoch_config:  # Empty list means test latest epoch
        return checkpoints[-1:] if checkpoints else []
        
    result = []
    total_epochs = len(checkpoints)
    
    for epoch in epoch_config:
        if epoch > 0:  # Positive number means specific epoch
            matches = [(path, num) for path, num in checkpoints if num == epoch]
            result.extend(matches)
        else:  # Negative number means count from end
            # Handle negative indices correctly
            index = total_epochs + epoch
            if 0 <= index < total_epochs:
                result.append(checkpoints[index])
                
    # Remove duplicates and sort
    unique_results = list({x[1]: x for x in result}.values())
    return sorted(unique_results, key=lambda x: x[1])

def get_valid_num_frames(num_frames: int) -> int:
    """Convert number of frames to valid format (4k + 1)
    
    Args:
        num_frames: Input number of frames
        
    Returns:
        int: Valid number of frames in format 4k + 1
    """
    if num_frames <= 1:
        return 1
    k = (num_frames - 1) / 4
    k = round(k)
    return 4 * k + 1

def generate_with_comfy_api(api_endpoint: str, payload: dict, output_path: str) -> None:
    logger.info(f"🚀 Preparing to send request to API: {api_endpoint}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")

    # Send main request
    start_time = time.time()
    response = requests.post(api_endpoint, json=payload, timeout=1200)
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        if content_type == 'text/plain':
            raise RuntimeError(f"API Error: {response.text}")
        ext_mapping = {
            'video/mp4': '.mp4', 'video/webm': '.webm', 'image/png': '.png',
            'image/gif': '.gif', 'image/jpeg': '.jpg', 'image/webp': '.webp'
        }
        ext = ext_mapping.get(content_type, '.mp4')  # Default to .mp4 if not specified
        output_path = os.path.splitext(output_path)[0] + ext
        with open(output_path, 'wb') as f:
            f.write(response.content)
        end_time = time.time()
        logger.info(f"✅ Success! File saved at: {output_path}, time: {end_time - start_time:.2f}s")
    else:
        raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
    
def get_model_local_path(url: str, model_type: ModelTypes) -> str:
    """Get local path for model based on its type and URL
    
    Args:
        url: HuggingFace URL
        model_type: Type of model (transformer, vae, clip, llm)
        
    Returns:
        str: Local path where model should be found after symlink
    """
    url_info = parse_hf_url(url)
    
    if model_type in [ModelTypes.TRANSFORMER, ModelTypes.VAE]:
        # For file-based models (transformer, vae)
        path = os.path.join(
            ModelPaths.TRAINING_BASE,
            TRAINING_PATHS[model_type],
            url_info["filename"]
        )
    else:
        # For folder-based models (clip, llm)
        repo_name = url_info["repo_id"].split("/")[-1]
        path = os.path.join(
            ModelPaths.TRAINING_BASE,
            TRAINING_PATHS[model_type],
            repo_name
        )
    
    # Convert Windows path separators to Unix style
    return path.replace('\\', '/')

def get_api_endpoint(endpoint_type: str = "comfy") -> str:
    """Get API endpoint from modal config
    
    Args:
        endpoint_type: Type of endpoint ("comfy", "hf_upload", or "hf_download")
        
    Returns:
        str: API endpoint URL if valid format, error message if not
    """
    try:
        username = modal.config._profile
        
        if endpoint_type == "comfy":
            return f"https://{username}--hy-training-comfyapi-endpoint.modal.run"
        elif endpoint_type == "hf_upload":
            return f"https://{username}--hy-training-upload.modal.run"
        elif endpoint_type == "hf_download":
            return f"https://{username}--hy-training-download.modal.run"
        else:
            return f"Invalid endpoint type: {endpoint_type}"
            
    except Exception as e:
        return f"Error getting API endpoint: {str(e)}"

def get_dataset_name(folder: str) -> str:
    """Get dataset name from dataset.toml in backup/config_orig folder
    
    Args:
        folder: Training folder name
        
    Returns:
        str: Dataset name or empty string if not found
    """
    try:
        volume = modal.Volume.from_name(Volumes.TRAINING)
        
        # Check if backup/config_orig/dataset.toml exists
        config_path = f"{folder}/backup/config_orig/dataset.toml"
        
        # List directory to check file existence
        dir_path = os.path.dirname(config_path)
        entries = volume.listdir(dir_path)
        
        if not any(entry.path.endswith("dataset.toml") for entry in entries):
            return "[no dataset info]"
            
        # Read and parse dataset.toml
        content = b"".join(volume.read_file(config_path))
        config = toml.loads(content.decode('utf-8'))
        
        # Extract dataset name from directory path
        if "directory" in config and len(config["directory"]) > 0:
            path = config["directory"][0].get("path", "")
            dataset = path.split("/")[-1]  # Get last part of path
            return f"[{dataset}]" if dataset else "[no dataset info]"
            
        return "[no dataset info]"
        
    except Exception:
        logger.warning(f"Not found dataset info for {folder}")
        return "[no dataset info]"

def get_epoch_info(folder: str) -> Dict[str, Any]:
    """Get epoch information from a training folder
    
    Args:
        folder: Training folder name
        
    Returns:
        Dict with keys: has_epochs, latest_epoch, epoch_list, total_epochs
    """
    try:
        processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
        epoch_info = processor._process_single_folder_epochs(folder)
        
        if not epoch_info:
            return {
                "has_epochs": False,
                "latest_epoch": 0,
                "epoch_list": [],
                "total_epochs": 0
            }
            
        return {
            "has_epochs": epoch_info.has_epochs,
            "latest_epoch": epoch_info.latest_epoch,
            "epoch_list": epoch_info.epoch_list,
            "total_epochs": epoch_info.total_epochs
        }
        
    except Exception as e:
        logger.error(f"Error getting epoch info for {folder}: {str(e)}")
        return {
            "has_epochs": False,
            "latest_epoch": 0,
            "epoch_list": [],
            "total_epochs": 0
        }

def get_step_info(folder: str) -> Dict[str, Any]:
    """Get global step information from a training folder
    
    Args:
        folder: Training folder name
        
    Returns:
        Dict with keys: has_steps, latest_step
    """
    try:
        processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
        step_info = processor._process_single_folder_steps(folder)
        
        if not step_info:
            return {
                "has_steps": False,
                "latest_step": 0
            }
            
        return {
            "has_steps": step_info.has_steps,
            "latest_step": step_info.latest_step
        }
        
    except Exception as e:
        logger.error(f"Error getting step info for {folder}: {str(e)}")
        return {
            "has_steps": False,
            "latest_step": 0
        }

def validate_api_endpoint(endpoint: str) -> bool:
    """Validate API endpoint format
    
    Args:
        endpoint: API endpoint URL to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    import re
    pattern = r'^https://[a-zA-Z0-9_-]+--hy-training-comfyapi-endpoint\.modal\.run$'
    return bool(re.match(pattern, endpoint))

def clear_caches():
    """Clear all caches to force reload from volumes"""
    with training_folders_cache_lock:
        training_folders_cache.clear()
    with epoch_cache_lock:
        epoch_cache.clear()
    with step_cache_lock:
        step_cache.clear()
    with test_outputs_cache_lock:
        test_outputs_cache.clear()
    with epochs_folders_cache_lock:
        epochs_folders_cache.clear()
    logger.info("All caches cleared")

def get_training_folders(force_reload: bool = False) -> List[str]:
    """Get list of training folders that contain test outputs
    
    Args:
        force_reload: If True, ignore cache and reload from volume
    
    Returns:
        List[str]: List of folder names with test output info
    """
    try:
        if not force_reload:
            with training_folders_cache_lock:
                if "folders" in training_folders_cache:
                    return training_folders_cache["folders"]
        
        volume = modal.Volume.from_name(Volumes.TRAINING)
        
        # List all entries
        entries = volume.listdir("/", recursive=False)
        
        # Filter folders and check for test outputs
        folders_with_outputs = []
        for entry in entries:
            if entry.type == FileEntryType.FILE:
                continue
                
            folder_name = entry.path.strip("/")
            
            # Check test_outputs folder
            try:
                test_outputs = volume.listdir(f"{folder_name}/test_outputs", recursive=False)
                media_files = [
                    f for f in test_outputs 
                    if f.path.lower().endswith(('.mp4', '.png', '.webp', '.jpeg', '.jpg'))
                ]
                
                if media_files:  # Only include folders with media files
                    # Get dataset name
                    dataset_name = get_dataset_name(folder_name)
                    
                    # Format folder name with file count and dataset
                    folder_info = f"{folder_name} ({len(media_files)} files) {dataset_name}"
                    folders_with_outputs.append(folder_info)
                    
            except Exception as e:
                logger.debug(f"No test outputs in {folder_name}: {str(e)}")
                continue
                
        result = sorted(folders_with_outputs, reverse=True) if folders_with_outputs else []
        
        with training_folders_cache_lock:
            training_folders_cache["folders"] = result
            
        return result
        
    except Exception as e:
        logger.error(f"Error loading training folders: {str(e)}")
        return []

def get_latest_training_folder():
    """Get latest training folder from volume"""
    folders = get_training_folders()
    return max(folders) if folders else None

def get_test_outputs(training_folder: str = None, force_reload: bool = False) -> List[Dict[str, Any]]:
    """Get test output files from a training folder with caching
    
    Args:
        training_folder: Name of the training folder to get outputs from
        force_reload: If True, bypass cache and reload from volume
        
    Returns:
        List of test output file information
    """
    if training_folder is None:
        return []
    
    with test_outputs_cache_lock:
        if training_folder in test_outputs_cache and not force_reload:
            return test_outputs_cache[training_folder]
    
    try:
        logger.info(f"Getting test outputs for folder: {training_folder}")
        volume = modal.Volume.from_name(Volumes.TRAINING)
        
        # Get training folder
        if not training_folder:
            training_folder = get_latest_training_folder()
            logger.info(f"Using latest training folder: {training_folder}")
        if not training_folder:
            logger.info("No training folder found")
            return []
            
        # Get test outputs folder
        test_outputs_path = f"{training_folder}/test_outputs"
        logger.info(f"Looking for test outputs in: {test_outputs_path}")
        
        # Create local directory for downloads if it doesn't exist
        local_cache_dir = os.path.join("cache", "test_outputs", training_folder)
        os.makedirs(local_cache_dir, exist_ok=True)
        logger.info(f"Using local cache directory: {local_cache_dir}")
        
        # List all files in test outputs
        try:
            entries = volume.listdir(test_outputs_path, recursive=False)
            logger.info(f"Found {len(entries)} entries in test outputs folder")
        except Exception as e:
            logger.error(f"Error listing directory {test_outputs_path}: {str(e)}")
            return []
            
        # Filter media files and parse info
        media_files = []
        for entry in entries:
            if entry.type != FileEntryType.FILE:
                continue
                
            # Get file extension from entry path
            _, ext = os.path.splitext(entry.path)
            ext = ext.lower()
            if ext not in ['.mp4', '.png', '.webp', '.jpeg', '.jpg']:
                continue
                
            # Parse filename from entry path
            remote_path = entry.path.lstrip("/")  # Remove leading slash
            filename = remote_path.split("/")[-1]  # Get last part of path
            logger.info(f"Processing file: {filename}")
            
            match = re.match(r'epoch(\d+)_prompt(\d+)_(\d{8}_\d{2}-\d{2}-\d{2}).*', filename)
            if not match:
                logger.warning(f"Filename {filename} doesn't match expected pattern")
                continue
                
            epoch = int(match.group(1))
            prompt_num = int(match.group(2))
            timestamp = datetime.strptime(match.group(3), '%Y%m%d_%H-%M-%S')
            
            # Download file to local cache if not exists
            local_path = os.path.join(local_cache_dir, filename)
            if not os.path.exists(local_path):
                logger.info(f"Downloading {remote_path} to {local_path}")
                try:
                    # Check file existence
                    dir_path = os.path.dirname(remote_path)
                    file_name = os.path.basename(remote_path)
                    entries = volume.listdir(dir_path)
                    file_exists = any(entry.path == remote_path for entry in entries)
                    if not file_exists:
                        logger.warning(f"File {remote_path} not found in volume")
                        continue
                    # Download file
                    with open(local_path, 'wb') as local_file:
                        for chunk in volume.read_file(remote_path):
                            local_file.write(chunk)
                except Exception as e:
                    logger.error(f"Download error: {str(e)}")
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    continue
            
            media_files.append({
                'path': local_path,
                'remote_path': remote_path,
                'type': 'video' if ext == '.mp4' else 'image',
                'epoch': epoch,
                'prompt_num': prompt_num,
                'timestamp': timestamp,
                'filename': filename
            })
            
        # Sort by epoch and prompt number
        sorted_files = sorted(media_files, key=lambda x: (x['epoch'], x['prompt_num']))
        logger.info(f"Returning {len(sorted_files)} media files")
        
        with test_outputs_cache_lock:
            test_outputs_cache[training_folder] = sorted_files
        
        return sorted_files
        
    except Exception as e:
        logger.error(f"Error in get_test_outputs: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_base_folder_name(folder: str) -> str:
    """Extract base folder name from formatted folder string
    
    Args:
        folder: Formatted folder name (can include dataset, epoch, step info)
        
    Returns:
        str: Base folder name or empty string
    """
    if not folder:
        return ""
    # Extract base folder name (format: YYYYMMDD_HH-MM-SS)
    return folder.split(" ")[0].split("-[")[0]

def get_folders_with_epochs(force_reload: bool = False) -> List[str]:
    """Get list of training folders that contain epoch folders
    
    Args:
        force_reload: If True, ignore cache and reload from volume
    
    Returns:
        List[str]: List of folder names with epoch info
    """
    try:
        if not force_reload:
            with epochs_folders_cache_lock:
                if "folders" in epochs_folders_cache:
                    return epochs_folders_cache["folders"]
        
        # Get list of valid training folders
        potential_folders = list_training_folders()
        
        if not potential_folders:
            return []
            
        # Process potential folders in parallel
        processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
        epoch_infos = processor.process_folders_parallel(potential_folders, process_type="epochs")
        
        # Format results
        formatted_folders = [
            f"{info.folder_name} {info.dataset_name} - "
            f"latest: epoch{info.latest_epoch} - "
            f"epoch[{', '.join(map(str, info.epoch_list))}] - "
            f"total: {info.total_epochs} epochs"
            for info in epoch_infos
        ]
        
        result = sorted(formatted_folders, reverse=True)
        
        with epochs_folders_cache_lock:
            epochs_folders_cache["folders"] = result
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting folders with epochs: {str(e)}")
        return []

def get_folders_with_steps(force_reload: bool = False) -> List[str]:
    """Get list of training folders that contain global_step folders
    
    Args:
        force_reload: If True, ignore cache and reload from volume
        
    Returns:
        List[str]: List of folder names with step info
    """
    try:
        if not force_reload:
            with step_cache_lock:
                if "folders" in step_cache:
                    return step_cache["folders"]
                    
        # Get list of valid training folders
        potential_folders = list_training_folders()
        
        if not potential_folders:
            return []
            
        # Process potential folders in parallel
        processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
        step_infos = processor.process_folders_parallel(potential_folders, process_type="steps")
        
        # Format results
        formatted_folders = [
            f"{info.folder_name} {info.dataset_name} - latest: step{info.latest_step}"
            for info in step_infos
        ]
        
        result = sorted(formatted_folders, reverse=True)
        
        with step_cache_lock:
            step_cache["folders"] = result
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting folders with steps: {str(e)}")
        return []

def get_formatted_folders(force_reload: bool = False) -> List[Tuple[str, str]]:
    """Get formatted list of folders with their display names using processor
    
    Args:
        force_reload: If True, bypass cache and reload from volume
    
    Returns:
        List[Tuple[str, str]]: List of (folder_name, display_name) tuples
    """
    try:
        processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
        
        # Get potential folders using common function
        potential_folders = list_training_folders()
        
        if not potential_folders:
            return []
            
        # Process folders in parallel using processor
        step_infos = processor.process_folders_parallel(potential_folders, process_type="steps", force_reload=force_reload)
        epoch_infos = processor.process_folders_parallel(potential_folders, process_type="epochs", force_reload=force_reload)
        
        # Create a map of folder name to epoch info
        epoch_info_map = {info.folder_name: info for info in epoch_infos if info}
        
        # Format results
        formatted_folders = []
        for step_info in step_infos:
            if not step_info:
                continue
                
            folder_name = step_info.folder_name
            epoch_info = epoch_info_map.get(folder_name)
            
            # Format display name
            display_name = f"{folder_name} {step_info.dataset_name}"
            
            if step_info.has_steps:
                display_name += f" - latest: step{step_info.latest_step}"
                
            if epoch_info and epoch_info.has_epochs:
                epoch_nums = sorted(epoch_info.epoch_list)
                display_name += f" - epochs[{','.join(map(str, epoch_nums))}]"
                
            formatted_folders.append((folder_name, display_name))
        
        return sorted(formatted_folders, key=lambda x: x[0], reverse=True)
        
    except Exception as e:
        logger.error(f"Error getting formatted folders: {str(e)}")
        return []

def update_epochs(folder: str) -> Dict:
    """Update epoch choices based on selected folder
    
    Args:
        folder: Selected training folder
        
    Returns:
        Dict: Gradio update for epoch dropdown
    """
    if not folder:
        return gr.update(choices=[], value=None)
        
    processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
    epoch_info = processor._process_single_folder_epochs(folder)
    
    if not epoch_info or not epoch_info.has_epochs:
        return gr.update(choices=[], value=None)
        
    epoch_choices = [f"epoch{e}" for e in sorted(epoch_info.epoch_list)]
    return gr.update(choices=epoch_choices, value=epoch_choices[0] if epoch_choices else None)

def update_steps(folder: str) -> Dict:
    """Update step choices based on selected folder
    
    Args:
        folder: Selected training folder
        
    Returns:
        Dict: Gradio update for step dropdown
    """
    if not folder:
        return gr.update(choices=[], value=None)
        
    processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
    step_info = processor._process_single_folder_steps(folder)
    
    if not step_info or not step_info.has_steps:
        return gr.update(choices=[], value=None)
        
    # Get all step folders
    volume = modal.Volume.from_name(Volumes.TRAINING)
    entries = volume.listdir(folder, recursive=False)
    steps = []
    for entry in entries:
        if entry.type == FileEntryType.FILE:
            continue
        step_name = os.path.basename(entry.path)
        if re.match(STEP_PATTERN, step_name):
            step_num = int(step_name.split("global_step")[1])
            steps.append(f"global_step{step_num}")
            
    step_choices = sorted(steps, key=lambda x: int(x.split("global_step")[1]))
    return gr.update(choices=step_choices, value=step_choices[0] if step_choices else None)

def get_latest_step_from_file(folder: str) -> Optional[int]:
    """Get latest step number from latest file
    
    Args:
        folder: Training folder name
        
    Returns:
        Optional[int]: Latest step number or None if not found
    """
    try:
        volume = modal.Volume.from_name(Volumes.TRAINING)
        latest_file = f"{folder}/latest"
        
        # Check if latest file exists
        entries = volume.listdir(folder, recursive=False)
        if not any(entry.path.endswith("/latest") for entry in entries):
            return None
            
        # Read and parse latest file
        content = b"".join(volume.read_file(latest_file))
        step_str = content.decode('utf-8').strip()
        
        # Extract step number
        if step_str.startswith("global_step"):
            try:
                return int(step_str[11:])  # Remove "global_step" prefix
            except ValueError:
                return None
                
        return None
        
    except Exception as e:
        logger.error(f"Error reading latest file in {folder}: {str(e)}")
        return None

def get_steps_from_folder(folder: str) -> List[str]:
    """Get list of global steps from training folder using processor
    
    Args:
        folder: Training folder path
        
    Returns:
        List[str]: List of global step folder names
    """
    try:
        latest_step = get_latest_step_from_file(folder)
        if latest_step is None:
            return []
            
        return [f"global_step{latest_step}"]
        
    except Exception as e:
        logger.error(f"Error getting steps from folder: {str(e)}")
        return []

def list_training_folders() -> List[str]:
    """Get list of valid training folders from volume
    
    Returns:
        List[str]: List of folder names matching training folder pattern
    """
    try:
        volume = modal.Volume.from_name(Volumes.TRAINING)
        entries = volume.listdir("/", recursive=False)
        
        # Filter folders using pattern matching
        folders = [
            entry.path.strip("/")
            for entry in entries
            if entry.type != FileEntryType.FILE and 
            re.match(FOLDER_PATTERN, entry.path.strip("/"))
        ]
        
        return sorted(folders, reverse=True)
        
    except Exception as e:
        logger.error(f"Error listing training folders: {str(e)}")
        return []

def get_tensorboard_folders(force_reload: bool = False) -> List[Tuple[str, str, List[str]]]:
    """Get list of training folders with log files and their latest log
    
    Args:
        force_reload: If True, ignore cache and reload from volume
        
    Returns:
        List[Tuple[str, str, List[str]]]: List of (display_name, folder_name, log_files) tuples
    """
    # Check cache first if not forcing reload
    cache_key = "tensorboard_folders"
    if not force_reload:
        with training_folders_cache_lock:
            if cache_key in training_folders_cache:
                return training_folders_cache[cache_key]
    
    try:
        volume = modal.Volume.from_name(Volumes.TRAINING)
        entries = volume.listdir("/", recursive=False)
        
        folder_data = []
        for entry in entries:
            if entry.type != FileEntryType.DIRECTORY:
                continue
                
            folder_name = entry.path.strip("/")
            try:
                all_entries = volume.listdir(folder_name, recursive=True)
                log_files = [e.path for e in all_entries if "events.out.tfevents" in e.path]
                
                if log_files:
                    # Get the latest log file
                    log_files.sort(reverse=True)
                    latest_log = log_files[0].split("/")[-1]
                    dataset_name = get_dataset_name(folder_name)
                    folder_data.append((
                        f"{folder_name} {dataset_name} (Logs: {len(log_files)})",
                        folder_name,
                        log_files
                    ))
                    
            except Exception as e:
                logger.warning(f"Skipped {folder_name}: {str(e)}")
                continue
                
        # Sort by folder name
        folder_data.sort(key=lambda x: x[1], reverse=True)
        
        # Update cache
        with training_folders_cache_lock:
            training_folders_cache[cache_key] = folder_data
            
        return folder_data
        
    except Exception as e:
        logger.error(f"Error getting TensorBoard folders: {str(e)}")
        return []