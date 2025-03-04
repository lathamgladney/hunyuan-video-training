from typing import Any, Dict
from _utils.constants import Config, Volumes, Paths, CheckpointModes, TestModes
import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

def get_default_config() -> Dict[str, Any]:
    """Get default configuration
    
    Returns:
        Dict[str, Any]: Default configuration
    """
    return {
        Config.Sections.CORE: {
            "gpu_type": "H100",
            "gpu_count": 1,
            "timeout_hours": 3,
            "gpu_type_test": "A100-40GB",
            "gpu_count_test": 1,
            "timeout_hours_test": 1,
            "volumes": [
                {"name": Volumes.TRAINING, "path": Paths.TRAINING.output},
                {"name": Volumes.CACHE, "path": Paths.CACHE.base},
                {"name": Volumes.COMFY, "path": Paths.INFERENCE.output}
            ]
        },
        
        Config.Sections.TRAINING: {
            # Basic settings
            "resume": False,
            "resume_folder": "",
            "dataset_dir": "dataset",
            
            # Training hyperparameters
            "epochs": 50,
            "micro_batch_size": 4,
            "gradient_accum_steps": 4,
            "warmup_steps": 100,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            
            # LoRA settings
            "lora_rank": 32,
            "lora_dtype": "bfloat16",
            
            # Checkpoint settings
            "save_every_n_epochs": 5,
            "checkpoint_mode": CheckpointModes.EPOCHS,
            "checkpoint_frequency": 5,
            
            # Advanced settings
            "pipeline_stages": 1,
            "gradient_clipping": 1.0,
            "activation_checkpointing": True,
            "eval_every_n_epochs": 5,
            "caching_batch_size": 4,
            "steps_per_print": 1,
            
            # Dataset settings
            "dataset": {
                "resolutions": [720],
                "enable_ar_bucket": True,
                "min_ar": 0.5,
                "max_ar": 2.0,
                "num_ar_buckets": 10,
                "frame_buckets": [8, 24],
                "num_repeats": 1
            }
        },
        
        Config.Sections.INFERENCE: {
            # ComfyUI settings
            "api_endpoint": "",
            "workflow_path": Config.Files.WORKFLOW,
            
            # Test settings
            "test_enabled": True,
            "test_strength": 0.85,
            "test_height": 320,
            "test_width": 576,
            "test_frames": 16,
            "test_steps": 25,
            
            # Test folder for testing
            "test_folder": "",
            
            # Test epoch selection
            "test_mode": TestModes.LATEST,
            "test_epochs": [],
            "test_latest_n": 1,
            
            # Test prompts
            "test_prompts": ["A beautiful sunset over the ocean"]
        },
        
        Config.Sections.MODELS: {
            # Model loading mode
            "use_official_ckpt": False,
            
            # Training models
            "training": {
                "transformer": "https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
                "vae": "https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main/hunyuan_video_vae_bf16.safetensors",
                "clip": "https://huggingface.co/openai/clip-vit-large-patch14/tree/main",
                "llm": "https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer/tree/main"
            },
            
            # Inference nodes and models
            "inference": {
                "nodes": [
                    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
                    "https://github.com/kijai/ComfyUI-HunyuanVideoWrapper"
                ],
                "specs": [
                    {
                        "link": "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_vae_bf16.safetensors",
                        "filename": "hunyuan_video_vae_bf16.safetensors",
                        "type": "vae"
                    },
                    {
                        "link": "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
                        "type": "unet"
                    },
                    {
                        "link": "https://huggingface.co/openai/clip-vit-large-patch14/tree/main",
                        "type": "clip"
                    },
                    {
                        "link": "https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer/tree/main",
                        "type": "LLM"
                    }
                ]
            }
        },
        
        Config.Sections.HF: {
            "auto_upload": False,
            "private_repo": True,
            "upload_test_outputs": True,
            "upload_tensorboard": False,
            "force_redownload": False,
            "skip_existing": True
        }
    }
