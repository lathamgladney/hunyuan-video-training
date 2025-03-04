# training.py
"""
Training logic for Hunyuan Video model
"""

import os
import glob
import re
from _utils.common_utils import (
    find_lora_checkpoints,
    get_valid_num_frames,
    generate_with_comfy_api,
)
from _utils.backup_utils import backup_training_files
from _utils.constants import Config, Paths
from _config import cfg
import time
from _shared.app import app, output_volume, cache_volume, config_volume, data_volume, nv_cache_volume, triton_cache_volume, inductor_cache_volume
from _utils.build_image import hunyuan_image
import modal
import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

@app.function(
    image=hunyuan_image,
    gpu=f"{cfg.core.gpu_type}:{cfg.core.gpu_count}",
    timeout=60*60*cfg.core.timeout_hours,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={
        str(Paths.TRAINING.output): output_volume,
        str(Paths.CACHE.base): cache_volume,
        "/root/config": config_volume,
        f"/root/{cfg.training.dataset_dir}": data_volume,  # Use dataset directory from config
        "/root/.nv": nv_cache_volume,
        "/root/.triton": triton_cache_volume,
        "/root/.inductor-cache": inductor_cache_volume,
    }
)
def remote_train():
    """Remote training function callable from Gradio"""
    os.chdir(Paths.TRAINING.base)

    # Get list of existing folders before training
    existing_folders = set()
    if not cfg.training.resume:
        existing_folders = set(glob.glob(os.path.join(Paths.TRAINING.output, "202*")))
    
    # Run training with deepspeed
    training_cmd = [
        "deepspeed",
        f"--num_gpus={cfg.core.gpu_count}",
        "train.py",
        "--deepspeed",
        "--config",
        f"{Paths.ROOT}/{Config.Files.HUNYUAN}"
    ]
    
    if cfg.training.resume:
        resume_folder = cfg.training.resume_folder
        if resume_folder:
            training_cmd.extend(["--resume_from_checkpoint", resume_folder])
            logger.info(f"Resuming training from folder: {resume_folder}")
        else:
            training_cmd.append("--resume_from_checkpoint")
            logger.info("Resuming training from latest checkpoint...")
    else:
        logger.info("Starting new training session...")
    
    logger.info("Starting training...")
    os.system(" ".join(training_cmd))
    
    # Get new training folder
    current_folders = set(glob.glob(os.path.join(Paths.TRAINING.output, "202*")))
    if not cfg.training.resume:
        new_folders = current_folders - existing_folders
        if not new_folders:
            logger.warning("No new training folder created")
            return
        new_folder = list(new_folders)[0]
    else:
        new_folder = max(current_folders, key=os.path.getctime)
        
    logger.info(f"Result training folder: {os.path.basename(new_folder)}")
    
    # Backup files
    backup_dir = os.path.join(new_folder, "backup")
    backup_training_files(
        backup_dir=backup_dir,
        config_src=Paths.TRAINING.config,
        dataset_src=f"{Paths.ROOT}/{cfg.training.dataset_dir}",
        dataset_name=cfg.training.dataset_dir,
        is_resume=cfg.training.resume
    )

    # Test LoRA models if enabled
    if cfg.inference.test_enabled:
        logger.info("Testing trained LoRA models...")
        test_frames = get_valid_num_frames(cfg.inference.test_frames)
        
        # Update test_folder logic based on training mode
        if cfg.training.resume:
            # For resume training, use the resume_folder if available
            test_folder = os.path.join(Paths.TRAINING.output, cfg.training.resume_folder)
            print(f"Resume training: Using test folder {test_folder}")
            logger.info(f"Resume training: Using test folder {test_folder}")
        else:
            # For new training, use the new_folder created in this session
            test_folder = new_folder
            print(f"New training: Using test folder {test_folder}")
            logger.info(f"New training: Using test folder {test_folder}")
            
        if not test_folder or not os.path.exists(test_folder):
            logger.error("No valid test folder found")
            return
        
        all_checkpoints = find_lora_checkpoints(test_folder)
        if not all_checkpoints:
            logger.error("No LoRA checkpoints found")
            return
        
        # Verify checkpoints exist and have sufficient size (at least 50MB)
        def verify_checkpoint(checkpoint_path, min_size_mb=50, max_attempts=10, wait_time=5):
            """
            Verify checkpoint exists and has sufficient size.
            
            Args:
                checkpoint_path: Path to checkpoint
                min_size_mb: Minimum size in MB
                max_attempts: Maximum number of retry attempts
                wait_time: Wait time in seconds between attempts
                
            Returns:
                bool: True if checkpoint is valid, False otherwise
            """
            min_size_bytes = min_size_mb * 1024 * 1024
            for attempt in range(max_attempts):
                if os.path.exists(checkpoint_path):
                    checkpoint_size = os.path.getsize(checkpoint_path)
                    if checkpoint_size >= min_size_bytes:
                        logger.info(f"Checkpoint {checkpoint_path} verified: {checkpoint_size/1024/1024:.2f}MB")
                        return True
                    else:
                        logger.warning(f"Checkpoint {checkpoint_path} found but size is only {checkpoint_size/1024/1024:.2f}MB, min required: {min_size_mb}MB. Attempt {attempt+1}/{max_attempts}")
                else:
                    logger.warning(f"Checkpoint {checkpoint_path} not found. Attempt {attempt+1}/{max_attempts}")
                
                if attempt < max_attempts - 1:
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
            
            return False
            
        # Filter verified checkpoints
        verified_checkpoints = []
        for checkpoint_path, epoch_num in all_checkpoints:
            if verify_checkpoint(checkpoint_path):
                verified_checkpoints.append((checkpoint_path, epoch_num))
            else:
                logger.error(f"Failed to verify checkpoint: {checkpoint_path}")
                
        if not verified_checkpoints:
            logger.error("No valid checkpoints found or all checkpoints are below minimum size")
            return
            
        checkpoints_to_test = verified_checkpoints[-1:]  # Default to latest verified checkpoint
        test_outputs_dir = os.path.join(test_folder, "test_outputs")
        os.makedirs(test_outputs_dir, exist_ok=True)
        
        api_endpoint = cfg.inference.api_endpoint
        if not api_endpoint:
            raise ValueError("api_endpoint is required")
        
        for checkpoint_path, epoch_num in checkpoints_to_test:
            logger.info(f"Testing LoRA from epoch {epoch_num}...")
            for i, prompt in enumerate(cfg.inference.test_prompts, 1):
                file_ext = "png" if test_frames == 1 else "mp4"
                output_path = os.path.join(
                    test_outputs_dir,
                    f"epoch{epoch_num:03d}_prompt{i:02d}_{time.strftime('%Y%m%d_%H-%M-%S')}_comfy_auto.{file_ext}"
                )
                rel_path = os.path.relpath(checkpoint_path, Paths.TRAINING.output)
                payload = {
                    "prompt": prompt,
                    "width": cfg.inference.test_width,
                    "height": cfg.inference.test_height,
                    "num_frames": test_frames,
                    "steps": cfg.inference.test_steps,
                    "strength": cfg.inference.test_strength,
                    "lora": rel_path
                }
                try:
                    generate_with_comfy_api(api_endpoint, payload, output_path)
                    logger.info(f"Generated test output for epoch {epoch_num}, prompt {i}")
                except Exception as e:
                    logger.error(f"Error generating test output: {str(e)}")