"""
Utility functions for downloading and managing models
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from huggingface_hub import snapshot_download, hf_hub_download
import modal
from .constants import (
    ModelPaths, ModelTypes, Paths,
    TRAINING_PATHS, COMFY_FOLDERS
)

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

def parse_hf_url(url: str) -> Dict:
    """Parse HuggingFace URL to get repo info"""
    try:
        # Handle empty or invalid URL
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid URL: {url}")
            
        # Check if URL is file or folder
        is_file = "/blob/" in url
        
        # Split URL into parts
        parts = url.split("/")
        if len(parts) < 5:
            raise ValueError(f"Invalid HuggingFace URL format: {url}")
            
        # Get repo_id
        repo_id = f"{parts[3]}/{parts[4]}"
        
        # Get branch and file_path for file URLs
        branch = "main"
        file_path = None
        filename = None
        
        if is_file:
            try:
                blob_index = parts.index("blob")
                if len(parts) > blob_index + 2:
                    branch = parts[blob_index + 1]
                    file_path = "/".join(parts[blob_index + 2:])
                    filename = parts[-1]
            except ValueError:
                logger.warning(f"Could not parse branch and file path from URL: {url}")
                
        return {
            "repo_id": repo_id,
            "is_file": is_file,
            "branch": branch,
            "file_path": file_path,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error parsing URL {url}: {str(e)}")
        raise

def create_symlink(source: str, target: str, is_dir: bool = False):
    """Create symlink with proper handling"""
    target_path = Path(target)
    
    # Create parent directory if needed
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing symlink/directory
    if target_path.exists():
        if target_path.is_dir():
            target_path.rmdir()
        else:
            target_path.unlink()
            
    # Create symlink
    os.symlink(source, target, target_is_directory=is_dir)
    logger.info(f"Created symlink: {target} -> {source}")

def download_hf_model(url_info: Dict, cache_path: str) -> Tuple[str, bool]:
    """Download model from HuggingFace and return path
    
    Args:
        url_info: Parsed URL info
        cache_path: Cache directory path
        
    Returns:
        Tuple[str, bool]: (downloaded_path, is_file)
    """
    if url_info["is_file"]:
        file_path = hf_hub_download(
            repo_id=url_info["repo_id"],
            filename=url_info["file_path"],
            cache_dir=Paths.CACHE.base,
            token=modal.Secret.from_name("huggingface-token"),
            revision=url_info["branch"]
        )
        return file_path, True
    else:
        snapshot_path = snapshot_download(
            repo_id=url_info["repo_id"],
            local_dir=cache_path,
            token=modal.Secret.from_name("huggingface-token"),
            revision=url_info["branch"]
        )
        
        # Get latest snapshot if exists
        snapshot_dir = os.path.join(snapshot_path, "snapshots")
        if os.path.exists(snapshot_dir):
            latest = sorted(os.listdir(snapshot_dir))[-1]
            snapshot_path = os.path.join(snapshot_dir, latest)
            
        return snapshot_path, False

def get_cache_path(repo_id: str) -> str:
    """Get HuggingFace cache path for repo"""
    repo_parts = repo_id.split("/")
    cache_folder = f"models--{repo_parts[0]}--{repo_parts[1]}"
    return os.path.join(Paths.CACHE.base, cache_folder)

def download_and_link_training_model(
    model_type: str,
    model_url: str
) -> bool:
    """Download training model and create appropriate symlink"""
    try:
        logger.info(f"Downloading {model_type} model from {model_url}")
        enum_model_type = ModelTypes(model_type)
        url_info = parse_hf_url(model_url)
        
        # Download model
        downloaded_path, is_file = download_hf_model(
            url_info, 
            get_cache_path(url_info["repo_id"])
        )
        
        # Create symlink based on model type
        if is_file:
            target_dir = os.path.join(ModelPaths.TRAINING_BASE, TRAINING_PATHS[enum_model_type])
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, url_info["filename"])
            create_symlink(downloaded_path, target)
            
            size_mb = os.path.getsize(target) / (1024 * 1024)
            logger.info(f"Downloaded {url_info['filename']} ({size_mb:.2f} MB)")
        else:
            repo_name = url_info["repo_id"].split("/")[-1]
            target = os.path.join(
                ModelPaths.TRAINING_BASE,
                TRAINING_PATHS[enum_model_type],
                repo_name
            )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            create_symlink(downloaded_path, target, is_dir=True)
            logger.info(f"Downloaded and linked {repo_name} folder")
                
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_type} model: {str(e)}")
        return False

def download_and_link_inference_model(
    model_url: str,
    model_type: str,
    custom_filename: Optional[str] = None
) -> bool:
    """Download inference model and create appropriate symlink"""
    try:
        enum_model_type = ModelTypes(model_type)
        logger.info(f"Downloading inference model {model_type} from {model_url}")
        url_info = parse_hf_url(model_url)
        
        # Download model
        downloaded_path, is_file = download_hf_model(
            url_info,
            get_cache_path(url_info["repo_id"])
        )
        
        # Create symlink based on model type
        if is_file:
            filename = custom_filename or url_info["filename"]
            target = os.path.join(
                ModelPaths.INFERENCE_BASE,
                COMFY_FOLDERS[enum_model_type],
                filename
            )
            create_symlink(downloaded_path, target)
            logger.info(f"Downloaded and linked {filename}")
        else:
            repo_name = url_info["repo_id"].split("/")[-1]
            target = os.path.join(
                ModelPaths.INFERENCE_BASE,
                COMFY_FOLDERS[enum_model_type],
                repo_name
            )
            create_symlink(downloaded_path, target, is_dir=True)
            logger.info(f"Downloaded and linked {repo_name} folder")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading inference model: {str(e)}")
        return False 