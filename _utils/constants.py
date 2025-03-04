"""
Constants for the Hunyuan Video project.
Organized into namespaces for different components.
"""

from typing import NamedTuple
from enum import Enum

# Application name used across the project
APP_NAME = "hy_training"

class PathConfig(NamedTuple):
    """Base path configuration for different components"""
    base: str
    config: str = None
    output: str = None

class ModelPaths:
    """Namespace for model paths"""
    # Training model paths
    TRAINING_BASE = "/root/diffusion-pipe/ckpts"
    TRAINING_TRANSFORMER = "hunyuan-video-t2v-720p/transformers"
    TRAINING_VAE = "hunyuan-video-t2v-720p/vae"
    TRAINING_CLIP = "text_encoder_2"
    TRAINING_LLM = "text_encoder"
    
    # Inference model paths (ComfyUI)
    INFERENCE_BASE = "/root/comfy/ComfyUI/models"

class Paths:
    """Namespace for all path-related constants"""
    
    # Base paths
    ROOT = "/root"
    
    # Training paths
    TRAINING = PathConfig(
        base="/root/diffusion-pipe",
        config="/root/config",
        output="/root/diffusion-pipe/hunyuan-video-lora"
    )
    
    # Checkpoint paths
    CKPTS = "/root/diffusion-pipe/ckpts"
    
    # Inference paths
    INFERENCE = PathConfig(
        base="/root/comfy",
        config="/root/config",
        output="/root/comfy/ComfyUI/output"
    )
    
    # Cache paths
    CACHE = PathConfig(
        base="/root/.cache/huggingface"
    )

class Volumes:
    """Namespace for volume names"""
    TRAINING = "hunyuan-training-results"
    CACHE = "hf-cache"
    COMFY = "comfy-output"
    CONFIG = "training-config"
    DATA = "training-data"
    
class Config:
    """Namespace for configuration constants"""
    
    class Files:
        """Configuration file paths"""
        MODAL = "config/modal.toml"
        DATASET = "config/dataset.toml"
        HUNYUAN = "config/hunyuan_video.toml"
        COMFY = "config/comfy_config.toml"
        WORKFLOW = "config/workflow_api.json"
    
    class Sections:
        """Configuration section names"""
        CORE = "core"
        TRAINING = "training"
        INFERENCE = "inference"
        MODELS = "models"
        HF = "hf"
        
    class Keys:
        """Configuration key names"""
        # Core keys
        GPU_TYPE = "gpu_type"
        GPU_COUNT = "gpu_count"
        TIMEOUT = "timeout_hours"
        VOLUMES = "volumes"
        
        # Test environment keys
        GPU_TYPE_TEST = "gpu_type_test"
        GPU_COUNT_TEST = "gpu_count_test" 
        TIMEOUT_TEST = "timeout_hours_test"
        
        # Training keys
        RESUME = "resume"
        RESUME_FOLDER = "resume_folder"
        DATASET_DIR = "dataset_dir"
        EPOCHS = "epochs"
        BATCH_SIZE = "micro_batch_size"
        
        # Test keys
        TEST_FOLDER = "test_folder"
        
        # Model keys
        MODEL_SPECS = "specs"
        
        # HF keys
        AUTO_UPLOAD = "auto_upload"
        PRIVATE_REPO = "private_repo"
        FORCE_REDOWNLOAD = "force_redownload"
        
    class Defaults:
        """Default configuration values"""
        GPU_TYPE = "A100-40GB"
        GPU_COUNT = 1
        TIMEOUT_HOURS = 3
        BATCH_SIZE = 8
        NUM_FRAMES = 73
        LORA_RANK = 32
        
class ModelTypes(str, Enum):
    """Model type constants"""
    TRANSFORMER = "transformer"
    VAE = "vae"
    CLIP = "clip"
    LLM = "llm"
    UNET = "unet"
    LORA = "lora"

# Model type properties - Moved outside of enum
MODEL_PROPERTIES = {
    ModelTypes.TRANSFORMER: {"is_file": True},
    ModelTypes.VAE: {"is_file": True},
    ModelTypes.CLIP: {"is_file": False},
    ModelTypes.LLM: {"is_file": False},
    ModelTypes.UNET: {"is_file": True},
    ModelTypes.LORA: {"is_file": True}
}

# Training model paths mapping - Moved outside of enum
TRAINING_PATHS = {
    ModelTypes.TRANSFORMER: ModelPaths.TRAINING_TRANSFORMER,
    ModelTypes.VAE: ModelPaths.TRAINING_VAE,
    ModelTypes.CLIP: ModelPaths.TRAINING_CLIP,
    ModelTypes.LLM: ModelPaths.TRAINING_LLM
}

# ComfyUI model folders mapping - Moved outside of enum
COMFY_FOLDERS = {
    ModelTypes.VAE: "vae",
    ModelTypes.UNET: "unet", 
    ModelTypes.CLIP: "clip",
    ModelTypes.LLM: "LLM",
    ModelTypes.LORA: "loras"
}
    
class CheckpointModes(str, Enum):
    """Namespace for checkpoint mode constants"""
    EPOCHS = "epochs"
    MINUTES = "minutes"
    
class TestModes:
    """Namespace for test mode constants"""
    LATEST = "latest"
    SPECIFIC = "specific"
    LATEST_N = "latest_n"
