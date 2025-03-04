# hf_operations.py
"""
HuggingFace operations (upload/download)
"""

import os
import modal
import glob
from typing import Optional
from huggingface_hub import HfApi, create_repo, snapshot_download
from _utils.constants import Paths
from _config import cfg
from _utils.build_image import hf_image
from _shared.app import app, output_volume

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

def get_latest_training_folder() -> str:
    folders = glob.glob(os.path.join(Paths.TRAINING.output, "[0-9]*_[0-9]*-[0-9]*-[0-9]*"))
    if not folders:
        return ""
    return sorted(folders, reverse=True)[0]

def get_latest_global_step(training_folder: str) -> str:
    latest_file = os.path.join(training_folder, "latest")
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            return f.read().strip()
    return ""

def get_latest_epoch(training_folder: str) -> str:
    epochs = glob.glob(os.path.join(training_folder, "epoch*"))
    if not epochs:
        return ""
    return sorted(epochs, key=lambda x: int(x.split("epoch")[-1]))[-1]

def upload_to_huggingface(path_or_fileobj: str, path_in_repo: str, repo_id: str, token: str, private: bool):
    api = HfApi()
    try:
        create_repo(repo_id=repo_id, private=private, token=token)
    except Exception:
        pass
    api.upload_file(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo, repo_id=repo_id, token=token)

@app.function(
    image=hf_image,
    volumes={Paths.TRAINING.output: output_volume},
    timeout=60*60*3,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
@modal.web_endpoint(method="POST")
def upload(repo: str, private: bool, include_tests: bool, target: str):
    logger.info(f"Uploading to HuggingFace repo: {repo} (private: {private}, include_tests: {include_tests})")
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        paths_to_upload = []
        training_folder = None

        if not target:
            raise ValueError("No target path provided")
            
        if "/" in target:
            base_folder, sub_path = target.split("/", 1)
            training_folder = os.path.join(Paths.TRAINING.output, base_folder)
            
            if "global_step" in sub_path and "epoch" in sub_path:
                step_part, epoch_part = sub_path.split(",")
                step_folder = os.path.join(training_folder, step_part)
                epoch_folder = os.path.join(training_folder, epoch_part)
                
                if os.path.exists(step_folder):
                    paths_to_upload.append(step_folder)
                if os.path.exists(epoch_folder):
                    paths_to_upload.append(epoch_folder)
                    
            elif "global_step" in sub_path:
                steps = sub_path.split("global_step")[1].split(",")
                for step in steps:
                    step_folder = os.path.join(training_folder, f"global_step{step}")
                    if os.path.exists(step_folder):
                        paths_to_upload.append(step_folder)
                            
            elif "epoch" in sub_path:
                epochs = sub_path.split("epoch")[1].split(",")
                for epoch in epochs:
                    epoch_folder = os.path.join(training_folder, f"epoch{epoch}")
                    if os.path.exists(epoch_folder):
                        paths_to_upload.append(epoch_folder)
                            
            else:
                full_path = os.path.join(Paths.TRAINING.output, target)
                if os.path.exists(full_path):
                    paths_to_upload.append(full_path)
                    
            if training_folder:
                for file in ["hunyuan_video.toml", "latest", "test_outputs"]:
                    file_path = os.path.join(training_folder, file)
                    if os.path.exists(file_path) and file_path not in paths_to_upload:
                        paths_to_upload.append(file_path)
        else:
            full_path = os.path.join(Paths.TRAINING.output, target)
            if os.path.isdir(full_path):
                paths_to_upload.append(full_path)

        uploaded_files = []
        for path in paths_to_upload:
            relative_path = os.path.relpath(path, Paths.TRAINING.output)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_relative_path = os.path.relpath(file_path, Paths.TRAINING.output)
                        upload_to_huggingface(file_path, file_relative_path, repo, hf_token, private)
                        uploaded_files.append(file_relative_path)
            else:
                upload_to_huggingface(path, relative_path, repo, hf_token, private)
                uploaded_files.append(relative_path)
                
        return {"status": "success", "message": "Upload completed", "uploaded_files": uploaded_files}
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

@app.function(
    image=hf_image,
    volumes={Paths.TRAINING.output: output_volume},
    timeout=60*60*3,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
@modal.web_endpoint(method="POST")
def download(repo: str, target: Optional[str] = None, force: bool = False):
    logger.info(f"Downloading from HuggingFace repo: {repo}")
    hf_token = os.getenv("HF_TOKEN")
    try:
        os.makedirs(Paths.TRAINING.output, exist_ok=True)
        
        # Parse target path
        patterns = []
        if target:
            if '/' in target:
                base_folder, sub_path = target.split('/', 1)
                if 'global_step' in sub_path and 'epoch' in sub_path:
                    step_part, epoch_part = sub_path.split(',')
                    patterns.append(f"{base_folder}/{step_part}/**")
                    patterns.append(f"{base_folder}/{epoch_part}/**")
                elif 'global_step' in sub_path:
                    steps = sub_path.split('global_step')[-1].split(',')
                    for step in steps:
                        patterns.append(f"{base_folder}/global_step{step}/**")
                elif 'epoch' in sub_path:
                    epochs = sub_path.split('epoch')[-1].split(',')
                    for epoch in epochs:
                        patterns.append(f"{base_folder}/epoch{epoch}/**")
                else:
                    patterns.append(f"{target}/**")
            else:
                patterns.append(f"{target}/**")
        else:
            patterns = ["**"]  # Download entire repo

        # Also include metadata files
        patterns += [
            f"{target.split('/')[0]}/hunyuan_video.toml",
            f"{target.split('/')[0]}/latest",
            # f"{target.split('/')[0]}/test_outputs/**"
        ] if target else []

        snapshot_download(
            repo_id=repo,
            local_dir=Paths.TRAINING.output,
            token=hf_token,
            force_download=force,
            allow_patterns=patterns,
            ignore_patterns=["*.bak"]
        )
        
        return {"status": "success", "message": "Download completed", "target": target or "latest"}
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
