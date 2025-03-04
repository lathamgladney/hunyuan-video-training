# _utils/config_normalizer.py
"""
Configuration normalizer for Hunyuan Video project
"""

import os
import toml
import logging
import shutil
from typing import Dict
from pathlib import Path
from .constants import Config

logger = logging.getLogger(__name__)

class ConfigNormalizer:
    def __init__(self):
        pass
    
    def normalize_config_paths(self):
        base_dir = Path(__file__).parent.parent
        config_dir = base_dir / "config"
        
        if not config_dir.exists():
            raise ValueError("Config directory not found")
        
        modal_config = self.load_and_validate_modal_config(base_dir / Config.Files.MODAL)
        self.generate_dataset_config(modal_config, base_dir / Config.Files.DATASET)
        self.generate_hunyuan_config(modal_config, base_dir / Config.Files.HUNYUAN)
        self.generate_comfy_config(modal_config, base_dir / Config.Files.COMFY)
    
    def load_and_validate_modal_config(self, config_path: Path) -> Dict:
        if not config_path.exists():
            raise ValueError("modal.toml not found")
        config = toml.load(str(config_path))
        required_sections = ["core", "training", "inference", "models"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing section: {section}")
        return config
    
    def generate_dataset_config(self, modal_config: Dict, config_path: Path):
        dataset_settings = modal_config["training"]["dataset"]
        dataset_config = {
            "resolutions": dataset_settings["resolutions"],
            "enable_ar_bucket": dataset_settings["enable_ar_bucket"],
            "min_ar": dataset_settings["min_ar"],
            "max_ar": dataset_settings["max_ar"],
            "num_ar_buckets": dataset_settings["num_ar_buckets"],
            "frame_buckets": dataset_settings["frame_buckets"],
            "directory": [{
                "path": "/root/data",  # Fixed path
                "num_repeats": dataset_settings["num_repeats"]
            }]
        }
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(dataset_config, f)
        logger.info("Generated dataset.toml")
    
    def generate_hunyuan_config(self, modal_config: Dict, config_path: Path):
        training_config = modal_config["training"]
        config = {
            "output_dir": "/root/diffusion-pipe/hunyuan-video-lora",
            "dataset": "/root/config/dataset.toml",
            "epochs": training_config["epochs"],
            "micro_batch_size_per_gpu": training_config["micro_batch_size"],
            "pipeline_stages": training_config["pipeline_stages"],
            "gradient_accumulation_steps": training_config["gradient_accum_steps"],
            "gradient_clipping": training_config["gradient_clipping"],
            "warmup_steps": training_config["warmup_steps"],
            "eval_every_n_epochs": training_config["eval_every_n_epochs"],
            "save_every_n_epochs": training_config["save_every_n_epochs"],
            "activation_checkpointing": training_config["activation_checkpointing"],
            "caching_batch_size": training_config["caching_batch_size"],
            "steps_per_print": training_config["steps_per_print"],
            "video_clip_mode": "single_middle",
            "checkpoint_every_n_epochs": training_config["checkpoint_frequency"],
            "model": {
                "type": "hunyuan-video",
                "dtype": "bfloat16",
                "transformer_dtype": "float8",
                "timestep_sample_method": "logit_normal",
                "transformer_path": training_config["models"]["training"]["transformer"],
                "vae_path": training_config["models"]["training"]["vae"],
                "clip_path": training_config["models"]["training"]["clip"],
                "llm_path": training_config["models"]["training"]["llm"]
            },
            "adapter": {
                "type": "lora",
                "rank": training_config["lora_rank"],
                "dtype": "bfloat16"
            },
            "optimizer": {
                "type": "adamw_optimi",
                "lr": training_config["learning_rate"],
                "weight_decay": training_config["weight_decay"],
                "betas": [0.9, 0.99],
                "eps": 1e-8
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        logger.info("Generated hunyuan_video.toml")
    
    def generate_comfy_config(self, modal_config: Dict, config_path: Path):
        inference_config = modal_config["models"]["inference"]
        config = {
            "nodes": {"nodes": inference_config["nodes"]},
            "models": inference_config["specs"]
        }
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("[nodes]\n")
            f.write('nodes = [\n  "' + '",\n  "'.join(config["nodes"]["nodes"]) + '",\n]\n\n')
            for model in config["models"]:
                f.write("[[models]]\n")
                for key, value in model.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f"{key} = {value}\n")
                f.write("\n")
        logger.info("Generated comfy_config.toml")