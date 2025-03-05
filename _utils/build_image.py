"""
Build and register Hunyuan and ComfyUI images
"""
from typing import List
from modal import Image
import modal
from pathlib import Path
import sys
import toml
from _utils.constants import Config, ModelTypes, Paths, Volumes
from _utils.model_utils import download_and_link_training_model, download_and_link_inference_model
import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)
# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)



def get_common_apt_packages() -> List[str]:
    """Get list of common apt packages"""
    return [
        "git",
        "wget",
        "curl",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    ]

def get_common_pip_packages() -> List[str]:
    """Get list of common pip packages"""
    return [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.48.2",
        "diffusers==0.32.2",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "huggingface_hub[hf_transfer]>=0.26.2",
        "pillow==11.1.0",
        "tqdm==4.67.1",
        "toml"
    ]

def get_common_env_vars() -> dict:
    """Get common environment variables"""
    return {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": Paths.CACHE.base,
        "HF_HUB_CACHE": Paths.CACHE.base,
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "PYTHONIOENCODING": "utf-8"
    }

def add_common_files(image: modal.Image) -> modal.Image:
    """Add common files to image"""
    return (image
        .add_local_python_source("_utils", "comfy_api", "comfy_ui", "_config", "main", "training", "hf_operations", "_shared", copy=True)
        .add_local_dir("config", remote_path="/root/config", copy=True)
    )

def _download_training_models() -> bool:
    """Download models for training"""
    try:
        logger.info("=== Starting _download_training_models ===")
        logger.info("Loading configuration...")
        config = toml.load(f"/root/{Config.Files.MODAL}")
        
        # Get training model config
        training_models = config["models"]["training"]
        
        # Download each model type
        for model_type in [ModelTypes.TRANSFORMER.value, ModelTypes.VAE.value, ModelTypes.CLIP.value, ModelTypes.LLM.value]:
            if model_type not in training_models:
                logger.warning(f"Missing {model_type} in training models config")
                continue
                
            model_url = training_models[model_type]
            logger.info(f"Processing model type: {model_type}")
            
            if not download_and_link_training_model(model_type, model_url):
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error downloading training models: {str(e)}", exc_info=True)
        return False

def _download_inference_models() -> bool:
    """Download models for inference"""
    try:
        logger.info("=== Starting _download_inference_models ===")
        logger.info("Loading configuration...")
        config = toml.load(f"/root/{Config.Files.COMFY}")
        
        for model_spec in config["models"]:
            try:
                model_url = model_spec["link"]
                model_type = model_spec["type"].lower()
                custom_filename = model_spec.get("filename")
                
                if not download_and_link_inference_model(model_url, model_type, custom_filename):
                    return False
                    
            except Exception as e:
                logger.error(f"Error processing model: {str(e)}")
                continue
                
        return True
        
    except Exception as e:
        logger.error(f"Error downloading inference models: {str(e)}")
        return False

# Build base image that will be inherited by both Hunyuan and ComfyUI
base_image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(get_common_apt_packages())
    .pip_install(get_common_pip_packages())
    .env(get_common_env_vars())
)

# Build HuggingFace image
hf_image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface-hub[hf_transfer]>=0.26.2",
        "fastapi[standard]>=0.115.4",
        "pydantic>=2.0.0",
        "toml>=0.10.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_image = add_common_files(hf_image)
# Build Hunyuan image
hunyuan_image = (
    base_image
    # Install Hunyuan specific packages
    .pip_install([
        "wheel==0.45.1",
        "protobuf==5.29.3",
        "loguru==0.7.3",
        "deepspeed==0.16.3",
        "datasets==3.2.0",
        "sentencepiece==0.2.0",
        "peft==0.14.0",
        "tensorboard==2.18.0",
        "bitsandbytes==0.45.1",
        "torch-optimi==0.2.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "av==14.1.0",
        "einops==0.8.0",
        "fastapi==0.111.0",
        "gradio"
    ])
    # Install flash-attn
    .run_commands([
        "pip install flash-attn==2.7.4.post1 --no-build-isolation"
    ])
    # Clone training repository
    .run_commands([
        "git clone --recurse-submodules https://github.com/AINxtGen/diffusion-pipe /root/diffusion-pipe",

    ],
    # force_build=True
    )
)

# Add common files and Python sources to Hunyuan image
hunyuan_image = (add_common_files(hunyuan_image)
    # Download training models
    .run_function(
        _download_training_models,
        volumes={
            "/root/.cache/huggingface": modal.Volume.from_name(Volumes.CACHE, create_if_missing=True)
        },
        secrets=[modal.Secret.from_name("huggingface-token")],
        force_build=True
    )
)

hunyuan_image = hunyuan_image.run_commands([
    "rm -rf /root/config 2>/dev/null || true"
])
# Build ComfyUI image
comfy_image = (
    base_image
    # Install ComfyUI specific packages
    .pip_install([
        "python-slugify==8.0.4",
        "python-dotenv==1.0.1",
        "python-multipart==0.0.9",
        "fastapi==0.111.0",
        "modal==0.67.43",
        "pydantic==2.7.3",
        "httpx==0.27.0",
        "uvicorn[standard]==0.25.0",
        "comfy-cli==1.3.7",
        "toml==0.10.2",
        "gradio"
    ])
    # Install ComfyUI
    .run_commands([
        "comfy --skip-prompt install --nvidia --version 0.3.18",
        f"rm -rf {Paths.INFERENCE.output} && mkdir -p {Paths.INFERENCE.output}"
    ])
)

# Add common files first to ensure config is available
comfy_image = add_common_files(comfy_image)

# Now install nodes using config that was just added
comfy_image = comfy_image.run_commands([
    # Install nodes from config
    'python3 -c "import toml; config = toml.load(\'/root/config/modal.toml\'); [print(f\'comfy node install {node}\') for node in config[\'models\'][\'inference\'][\'nodes\']]" | bash'
])

# Download inference models
comfy_image = comfy_image.run_function(
    _download_inference_models,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(Volumes.CACHE, create_if_missing=True)
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    force_build=True
)

comfy_image = comfy_image.run_commands([
    "rm -rf /root/config 2>/dev/null || true"
])
