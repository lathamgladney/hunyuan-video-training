# comfy_ui.py
"""
ComfyUI web interface
"""

import subprocess
import modal
from _utils.constants import Paths
import os
from _shared.app import app, output_volume, cache_volume, config_volume, comfy_output_vol
from _utils.build_image import comfy_image
from _config import cfg

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

@app.function(
    image=comfy_image,
    allow_concurrent_inputs=10,
    container_idle_timeout=300,
    timeout=900,
    gpu=f"{cfg.core.gpu_type_test}:{cfg.core.gpu_count_test}",
    volumes={
        Paths.CACHE.base: cache_volume,
        Paths.INFERENCE.output: comfy_output_vol,
        Paths.TRAINING.output: output_volume,
        Paths.TRAINING.config: config_volume
    }
)
@modal.web_server(8000, startup_timeout=300)
def comfy_ui():
    """Start ComfyUI web interface"""
    training_results_dir = Paths.TRAINING.output
    loras_base_dir = f"{Paths.INFERENCE.base}/ComfyUI/models/loras"
    
    if os.path.exists(training_results_dir):
        for root, dirs, files in os.walk(training_results_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    src_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, training_results_dir)
                    dest_dir = os.path.join(loras_base_dir, relative_path)
                    dest_path = os.path.join(dest_dir, file)
                    os.makedirs(dest_dir, exist_ok=True)
                    if not os.path.exists(dest_path):
                        os.symlink(src_path, dest_path)
                        logger.info(f"Created symlink: {dest_path}")
                    elif os.path.realpath(dest_path) != os.path.realpath(src_path):
                        os.remove(dest_path)
                        os.symlink(src_path, dest_path)
                        logger.info(f"Updated symlink: {dest_path}")

    models_dir = f"{Paths.INFERENCE.base}/ComfyUI/models"
    logger.info(f"Checking models directory: {models_dir}")
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                if f.endswith(('.safetensors', '.pt', '.cks')):
                    file_path = os.path.join(root, f)
                    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
                    logger.info(f"File: {f}, Size: {file_size_gb:.2f} GB, Path: {file_path}")

    custom_nodes_dir = f"{Paths.INFERENCE.base}/ComfyUI/custom_nodes"
    logger.info(f"Checking custom nodes directory: {custom_nodes_dir}")
    if os.path.exists(custom_nodes_dir):
        for root, dirs, _ in os.walk(custom_nodes_dir):
            for d in dirs:
                logger.info(f"Node: {d}")
            break
        
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000",
        shell=True,
        env={"PYTHONIOENCODING": "utf-8"}
    )