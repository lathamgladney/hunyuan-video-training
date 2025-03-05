import threading
import gradio as gr
import toml
from pathlib import Path
import os
from typing import Dict, Any, List, Optional
from _utils.common_utils import (
    get_api_endpoint, get_base_folder_name, get_folders_with_epochs,
    get_folders_with_steps, get_formatted_folders, get_model_local_path,
    get_test_outputs, get_training_folders, get_valid_num_frames,
    upload_config_files, upload_dataset, generate_with_comfy_api, FolderProcessor, get_tensorboard_folders, get_epochs_to_test
)
from _utils.config_normalizer import ConfigNormalizer
from _utils.constants import Config, ModelTypes, CheckpointModes, TestModes, Paths, Volumes, APP_NAME
import modal
import requests
import json
from _config import load_config
from _utils.default_config import get_default_config
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import atexit
import logging
from _utils.logging_config import configure_logging
import re
import subprocess

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

config_normalizer = ConfigNormalizer()
TEST_EXECUTOR = ThreadPoolExecutor(max_workers=1)
TEST_FUTURES = []
TEST_COUNTER = 0

@atexit.register
def graceful_shutdown():
    for future in TEST_FUTURES:
        if not future.done():
            future.cancel()
    TEST_EXECUTOR.shutdown(wait=True, cancel_futures=True)

def init_config_files():
    os.makedirs("config", exist_ok=True)
    required_files = [Config.Files.MODAL, Config.Files.DATASET, Config.Files.HUNYUAN, Config.Files.COMFY]
    for file_path in required_files:
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("")

def load_config(config_path: str) -> Dict[str, Any]:
    init_config_files()
    if not os.path.exists(config_path):
        config = get_default_config()
        save_config(config, config_path)
        return config
    config = toml.load(config_path)
    if not config:
        config = get_default_config()
        save_config(config, config_path)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)

class ConfigSettings:
    def __init__(self):
        self.epochs = 0
        self.micro_batch_size = 0
        self.gradient_accum_steps = 0
        self.warmup_steps = 0
        self.learning_rate = 0.0
        self.weight_decay = 0.0
        self.resume_training = False
        self.resume_folder = ""
        self.dataset_dir = ""
        self.gpu_type = ""
        self.gpu_count = 0
        self.timeout_hours = 0
        self.gpu_type_test = ""
        self.gpu_count_test = 0
        self.timeout_hours_test = 0
        self.enable_ar_bucket = False
        self.resolutions = 0
        self.min_ar = 0.0
        self.max_ar = 0.0
        self.num_ar_buckets = 0
        self.frame_buckets_min = 0
        self.frame_buckets_max = 0
        self.num_repeats = 0
        self.lora_rank = 0
        self.test_strength = 0.0
        self.test_enabled = False
        self.test_height = 0
        self.test_width = 0
        self.test_frames = 0
        self.test_steps = 0
        self.test_prompts = []
        self.test_epoch_mode = ""
        self.test_specific_epochs = ""
        self.test_latest_n_epochs = 0
        self.test_folder = ""
        self.api_endpoint = ""
        self.activation_checkpointing = False
        self.gradient_clipping = 0.0
        self.eval_every_n_epochs = 0
        self.save_every_n_epochs = 0
        self.pipeline_stages = 0
        self.checkpoint_mode = ""
        self.checkpoint_every_n_epochs = 0
        self.checkpoint_every_n_minutes = 0
        self.caching_batch_size = 0
        self.steps_per_print = 0
        self.transformer_path = ""
        self.vae_path = ""
        self.llm_path = ""
        self.clip_path = ""
        self.private_check = False
        self.test_outputs_check = False
        self.upload_endpoint = ""
        self.download_endpoint = ""
        self.repo_upload = ""

def update_configs(settings: ConfigSettings) -> str:
    try:
        clean_resume_folder = get_base_folder_name(settings.resume_folder)
        clean_test_folder = get_base_folder_name(settings.test_folder)
        valid_num_frames = get_valid_num_frames(int(settings.test_frames))
        
        modal_config = {
            Config.Sections.CORE: {
                "gpu_type": settings.gpu_type,
                "gpu_count": settings.gpu_count,
                "timeout_hours": settings.timeout_hours,
                "gpu_type_test": settings.gpu_type_test,
                "gpu_count_test": settings.gpu_count_test,
                "timeout_hours_test": settings.timeout_hours_test,
                "volumes": [
                    {"name": Volumes.TRAINING, "path": Paths.TRAINING.output},
                    {"name": Volumes.CACHE, "path": Paths.CACHE.base},
                    {"name": Volumes.COMFY, "path": Paths.INFERENCE.output}
                ]
            },
            Config.Sections.TRAINING: {
                "resume": settings.resume_training,
                "resume_folder": clean_resume_folder,
                "dataset_dir": settings.dataset_dir,
                "epochs": settings.epochs,
                "micro_batch_size": settings.micro_batch_size,
                "gradient_accum_steps": settings.gradient_accum_steps,
                "warmup_steps": settings.warmup_steps,
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "lora_rank": settings.lora_rank,
                "lora_dtype": "bfloat16",
                "save_every_n_epochs": settings.save_every_n_epochs,
                "checkpoint_mode": settings.checkpoint_mode,
                "checkpoint_frequency": (
                    settings.checkpoint_every_n_epochs
                    if settings.checkpoint_mode == CheckpointModes.EPOCHS
                    else settings.checkpoint_every_n_minutes
                ),
                "pipeline_stages": settings.pipeline_stages,
                "gradient_clipping": settings.gradient_clipping,
                "activation_checkpointing": settings.activation_checkpointing,
                "eval_every_n_epochs": settings.eval_every_n_epochs,
                "caching_batch_size": settings.caching_batch_size,
                "steps_per_print": settings.steps_per_print,
                "dataset": {
                    "resolutions": [settings.resolutions],
                    "enable_ar_bucket": settings.enable_ar_bucket,
                    "min_ar": settings.min_ar,
                    "max_ar": settings.max_ar,
                    "num_ar_buckets": settings.num_ar_buckets,
                    "frame_buckets": [settings.frame_buckets_min, settings.frame_buckets_max],
                    "num_repeats": settings.num_repeats
                }
            },
            Config.Sections.INFERENCE: {
                "api_endpoint": settings.api_endpoint,
                "workflow_path": Config.Files.WORKFLOW,
                "test_enabled": settings.test_enabled,
                "test_strength": settings.test_strength,
                "test_height": settings.test_height,
                "test_width": settings.test_width,
                "test_frames": valid_num_frames,
                "test_steps": settings.test_steps,
                "test_folder": clean_test_folder,
                "test_mode": settings.test_epoch_mode,
                "test_epochs": (
                    [int(e.strip()) for e in settings.test_specific_epochs.split(",") if e.strip()]
                    if settings.test_epoch_mode == TestModes.SPECIFIC else []
                ),
                "test_latest_n": (
                    settings.test_latest_n_epochs
                    if settings.test_epoch_mode == TestModes.LATEST_N else 1
                ),
                "test_prompts": settings.test_prompts.split("\n") if settings.test_prompts else []
            },
            Config.Sections.MODELS: {
                "training": {
                    "transformer": settings.transformer_path,
                    "vae": settings.vae_path,
                    "clip": settings.clip_path,
                    "llm": settings.llm_path
                },
                "inference": {
                    "nodes": [
                        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
                        "https://github.com/kijai/ComfyUI-HunyuanVideoWrapper"
                    ],
                    "specs": [
                        {"link": "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_vae_bf16.safetensors", "filename": "hunyuan_video_vae_bf16.safetensors", "type": "vae"},
                        {"link": "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", "type": "unet"},
                        {"link": "https://huggingface.co/openai/clip-vit-large-patch14/tree/main", "type": "clip"},
                        {"link": "https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer/tree/main", "type": "LLM"}
                    ]
                }
            },
            Config.Sections.HF: {
                "auto_upload": False,
                "private_repo": settings.private_check,
                "upload_test_outputs": settings.test_outputs_check,
                "upload_tensorboard": False,
                "force_redownload": False,
                "skip_existing": True,
                "upload_endpoint": settings.upload_endpoint,
                "download_endpoint": settings.download_endpoint,
                "repo_upload": settings.repo_upload
            }
        }
        with open(Config.Files.MODAL, "w", encoding="utf-8") as f:
            toml.dump(modal_config, f)
            logger.info("Saved to config/modal.toml")

        hunyuan_config = {
            "output_dir": "/root/diffusion-pipe/hunyuan-video-lora",
            "dataset": "/root/config/dataset.toml",
            "epochs": settings.epochs,
            "micro_batch_size_per_gpu": settings.micro_batch_size,
            "pipeline_stages": settings.pipeline_stages,
            "gradient_accumulation_steps": settings.gradient_accum_steps,
            "gradient_clipping": settings.gradient_clipping,
            "warmup_steps": settings.warmup_steps,
            "eval_every_n_epochs": settings.eval_every_n_epochs,
            "eval_before_first_step": True,
            "eval_micro_batch_size_per_gpu": 1,
            "eval_gradient_accumulation_steps": 1,
            "save_every_n_epochs": settings.save_every_n_epochs,
            "activation_checkpointing": settings.activation_checkpointing,
            "partition_method": "parameters",
            "save_dtype": "bfloat16",
            "caching_batch_size": settings.caching_batch_size,
            "steps_per_print": settings.steps_per_print,
            "video_clip_mode": "single_middle",
            "checkpoint_every_n_epochs": settings.checkpoint_every_n_epochs,
            "model": {
                "type": "hunyuan-video",
                "dtype": "bfloat16",
                "transformer_dtype": "float8",
                "timestep_sample_method": "logit_normal",
                "transformer_path": get_model_local_path(settings.transformer_path, ModelTypes.TRANSFORMER),
                "vae_path": get_model_local_path(settings.vae_path, ModelTypes.VAE),
                "clip_path": get_model_local_path(settings.clip_path, ModelTypes.CLIP),
                "llm_path": get_model_local_path(settings.llm_path, ModelTypes.LLM)
            },
            "adapter": {
                "type": "lora",
                "rank": settings.lora_rank,
                "dtype": "bfloat16"
            },
            "optimizer": {
                "type": "adamw_optimi",
                "lr": settings.learning_rate,
                "betas": [0.9, 0.99],
                "weight_decay": settings.weight_decay,
                "eps": 1e-8
            }
        }

        with open(Config.Files.HUNYUAN, "w", encoding="utf-8") as f:
            toml.dump(hunyuan_config, f)

        dataset_config = {
            "resolutions": [settings.resolutions],
            "enable_ar_bucket": settings.enable_ar_bucket,
            "min_ar": settings.min_ar,
            "max_ar": settings.max_ar,
            "num_ar_buckets": settings.num_ar_buckets,
            "frame_buckets": [settings.frame_buckets_min, settings.frame_buckets_max],
            "directory": [{"path": "/root/data", "num_repeats": settings.num_repeats}]
        }

        with open(Config.Files.DATASET, "w", encoding="utf-8") as f:
            toml.dump(dataset_config, f)

        return "✅ [SUCCESS] All configurations saved successfully!"
    except Exception as e:
        return f"⚠️ Error saving configurations: {str(e)}"

def start_training(settings: ConfigSettings) -> str:
    save_result = update_configs(settings)
    if save_result.startswith("⚠️"):
        return save_result
    try:
        from modal import Function
        config = load_config(Config.Files.MODAL)
        training = Function.from_name(APP_NAME, "remote_train")
        config_volume = modal.Volume.from_name(Volumes.CONFIG, create_if_missing=True)
        dataset_dir = config[Config.Sections.TRAINING]["dataset_dir"]
        data_volume = modal.Volume.from_name(Volumes.DATA, create_if_missing=True)
        upload_config_files(config_volume, "config")
        upload_dataset(data_volume, dataset_dir)
        training.spawn()
        return "✅ Training request sent to Modal! Check logs: https://modal.com/logs"
    except Exception as e:
        return f"⚠️ Error starting training: {str(e)}"

def scan_dataset_directories() -> List[str]:
    """
    Scan the current directory for folders containing images or videos at level 1 only.
    Returns a list of folder choices with format: "folder_name (X images, Y videos, Z captions)"
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    dataset_choices = []
    root_dir = os.getcwd()
    
    # Get all directories in the current working directory
    for item in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, item)
        if os.path.isdir(dir_path):
            image_count = 0
            video_count = 0
            caption_count = 0
            
            # Count files ONLY at level 1 (not recursively)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):  # Only count files, not directories
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in image_extensions):
                        image_count += 1
                    elif any(file_lower.endswith(ext) for ext in video_extensions):
                        video_count += 1
                    elif file_lower.endswith('.txt'):
                        caption_count += 1
            
            # Only include directories with at least one image or video
            if image_count > 0 or video_count > 0:
                display_name = f"{item} ({image_count} images, {video_count} videos"
                if caption_count > 0:
                    display_name += f", {caption_count} captions"
                display_name += ")"
                
                dataset_choices.append((display_name, dir_path))
    
    # Sort choices alphabetically by directory name
    dataset_choices.sort(key=lambda x: x[0])
    
    return dataset_choices

def update_dataset_dropdown():
    """
    Returns information for the dataset directory dropdown
    If only one valid directory is found, automatically select it
    """
    choices = scan_dataset_directories()
    
    # Convert to the format expected by gr.Dropdown
    dropdown_choices = [choice[0] for choice in choices]
    
    # If no directories found
    if not dropdown_choices:
        return gr.update(choices=[], value=None)
    
    # Only auto-select if there's exactly one directory
    selected_value = dropdown_choices[0] if len(choices) == 1 else None
    
    return gr.update(choices=dropdown_choices, value=selected_value)

def extract_dir_path_from_display(display_text: str) -> str:
    """
    Extract just the folder name from the display text shown in dropdown
    """
    # Check if input is a list (can happen with Gradio Dropdown)
    if isinstance(display_text, list):
        # If it's a list with elements, use the first element
        if display_text:
            display_text = display_text[0]
        else:
            return ""
    
    # The display format is "folder_name (X images, Y videos, Z captions)"
    # We need to extract just the folder_name part
    if not display_text:
        return ""
    
    # Extract the folder name (everything before the first opening parenthesis)
    if isinstance(display_text, str) and " (" in display_text:
        folder_name = display_text.split(" (")[0]
    else:
        # If custom value entered, use as is
        folder_name = display_text
    
    # If it's already a full path, extract just the folder name
    if isinstance(folder_name, str) and os.path.isabs(folder_name):
        return os.path.basename(folder_name)
        
    # Return just the folder name
    return str(folder_name)

def on_dataset_selected(display_text: str) -> None:
    """
    Process the selected dataset directory and update config
    """
    if not display_text:
        return
    
    # Handle case when display_text is a list
    if isinstance(display_text, list):
        if not display_text:
            return
        display_text = display_text[0]
        
    # Extract the actual directory path
    dir_path = extract_dir_path_from_display(display_text)
    
    # Update config with the selected directory
    config = load_config(Config.Files.MODAL)
    config[Config.Sections.TRAINING]["dataset_dir"] = dir_path
    save_config(config, Config.Files.MODAL)

def update_resume_folders():
    formatted_folders = get_formatted_folders(force_reload=True)
    folders = [display for _, display in formatted_folders]
    latest = folders[0] if folders else None
    return gr.update(choices=folders, value=latest)

def update_test_folders():
    folders = get_folders_with_epochs(force_reload=True)
    latest = folders[0] if folders else None
    return gr.update(choices=folders, value=latest)

def update_training_folders():
    folders = get_training_folders(force_reload=True)
    latest = folders[0] if folders else None
    return gr.update(choices=folders, value=latest)

def update_results(folder: str = None, force_reload: bool = False):
    if not folder:
        return None
    actual_folder = folder.split(" (")[0]
    files = get_test_outputs(actual_folder, force_reload=force_reload)
    if not files:
        return None
    return [(file['path'], f"Epoch {file['epoch']:03d} - Prompt {file['prompt_num']:02d} - \n{file['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}") for file in files]

def update_api_endpoint():
    endpoint = get_api_endpoint("comfy")
    return gr.update(value=endpoint)

def save_config_and_start_training(settings: ConfigSettings) -> str:
    return start_training(settings)

def save_config_and_start_test_lora(settings: ConfigSettings) -> str:
    global TEST_COUNTER, TEST_FUTURES
    
    save_result = update_configs(settings)
    if save_result.startswith("⚠️"):
        return save_result
    
    api_endpoint = get_api_endpoint("comfy")
    base_folder = get_base_folder_name(settings.test_folder)
    processor = FolderProcessor(modal.Volume.from_name(Volumes.TRAINING))
    epoch_info = processor._process_single_folder_epochs(base_folder)
    if not epoch_info or not epoch_info.has_epochs:
        return f"⚠️ No epochs found in folder {base_folder}"
    
    # Get all available epochs and create checkpoints list
    checkpoints = [(base_folder, e) for e in epoch_info.epoch_list]
    
    # Prepare epoch config based on test mode
    if settings.test_epoch_mode == "specific" and settings.test_specific_epochs:
        epoch_config = [int(e.strip()) for e in settings.test_specific_epochs.split(",")]
    elif settings.test_epoch_mode == "latest_n" and settings.test_latest_n_epochs:
        # Use negative indices to select last N epochs
        epoch_config = [-i for i in range(1, int(settings.test_latest_n_epochs)+1)]
    else:
        epoch_config = []  # Will default to latest
    
    # Get actual epochs to test using common utility function
    selected = get_epochs_to_test(checkpoints, epoch_config)
    epochs_to_test = [epoch for _, epoch in selected]
    
    prompts = [p.strip() for p in settings.test_prompts.split("\n")] if settings.test_prompts else ["A beautiful video, best quality"]
    
    file_ext = "png" if settings.test_frames == 1 else "mp4"
    for epoch in epochs_to_test:
        logger.info(f"Testing epoch {epoch}")
        for prompt_idx, prompt in enumerate(prompts):
            TEST_COUNTER += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
            output_path = os.path.join("cache", "test_outputs", base_folder, f"epoch{epoch:03d}_prompt{prompt_idx+1:02d}_{timestamp}_comfy_manual.{file_ext}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            payload = {
                "strength": settings.test_strength,
                "width": settings.test_width,
                "height": settings.test_height,
                "num_frames": settings.test_frames,
                "steps": settings.test_steps,
                "prompt": prompt,
                "lora": f"{base_folder}/epoch{epoch}/adapter_model.safetensors"
            }
            future = TEST_EXECUTOR.submit(lambda p=payload, o=output_path: generate_with_comfy_api(api_endpoint, p, o))
            TEST_FUTURES.append(future)
    
    TEST_FUTURES = [f for f in TEST_FUTURES if not f.done()]
    total_tasks = len(epochs_to_test) * len(prompts)
    running = sum(1 for f in TEST_FUTURES if f.running())
    pending = sum(1 for f in TEST_FUTURES if not f.done())
    return f"✅ Queued {total_tasks} tests\nRunning: {running}, Pending: {pending}"

def parse_folder_display(folder_display: str) -> dict:
    try:
        return {'base_folder': folder_display.split()[0], 'full_name': folder_display.split(' [')[0]}
    except Exception as e:
        return {'base_folder': folder_display, 'full_name': folder_display}

def update_repo_folders(repo_id: str):
    """Get folder list from HuggingFace Hub"""
    from huggingface_hub import HfApi
    
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)
        
        # Process folder structure
        folders = {}
        for file in files:
            if '/' in file:
                folder = file.split('/')[0]
                if folder not in folders:
                    folders[folder] = {'steps': set(), 'epochs': set()}
                
                # Detect steps and epochs
                if 'global_step' in file:
                    step = file.split('global_step')[-1].split('/')[0]
                    if step.isdigit():
                        folders[folder]['steps'].add(int(step))
                elif 'epoch' in file and not file.endswith('.toml'):
                    epoch = file.split('epoch')[-1].split('/')[0]
                    if epoch.isdigit():
                        folders[folder]['epochs'].add(int(epoch))
        
        # Format display
        formatted = []
        for folder, data in folders.items():
            steps = sorted(data['steps'])
            epochs = sorted(data['epochs'])
            step_str = f"steps[{','.join(map(str, steps))}]" if steps else ""
            epoch_str = f"epochs[{','.join(map(str, epochs))}]" if epochs else ""
            display = f"{folder} - {step_str} - {epoch_str}"
            formatted.append((display, folder))
        
        return gr.update(choices=[f[0] for f in formatted], value=formatted[0][0] if formatted else None)
    
    except Exception as e:
        logger.error(f"Error fetching repo folders: {str(e)}")
        return gr.update(choices=[], value=None)

def gather_settings(
    epochs, micro_batch_size, gradient_accum_steps, warmup_steps, learning_rate,
    weight_decay, resume_training, resume_folder, dataset_dir, gpu_type,
    gpu_count, timeout_hours, gpu_type_test, gpu_count_test, timeout_hours_test,
    enable_ar_bucket, resolutions, min_ar, max_ar, num_ar_buckets,
    frame_buckets_min, frame_buckets_max, num_repeats, lora_rank, test_strength,
    test_enabled, test_height, test_width, test_frames, test_steps, test_prompts,
    test_epoch_mode, test_specific_epochs, test_latest_n_epochs, test_folder,
    api_endpoint, activation_checkpointing, gradient_clipping, eval_every_n_epochs,
    save_every_n_epochs, pipeline_stages, checkpoint_mode, checkpoint_every_n_epochs,
    checkpoint_every_n_minutes, caching_batch_size, steps_per_print, transformer_path,
    vae_path, llm_path, clip_path, private_check, test_outputs_check, upload_endpoint,
    download_endpoint, repo_upload
):
    settings = ConfigSettings()
    settings.epochs = epochs
    settings.micro_batch_size = micro_batch_size
    settings.gradient_accum_steps = gradient_accum_steps
    settings.warmup_steps = warmup_steps
    settings.learning_rate = learning_rate
    settings.weight_decay = weight_decay
    settings.resume_training = resume_training
    settings.resume_folder = resume_folder
    
    # Process dataset_dir from dropdown
    if isinstance(dataset_dir, list) and dataset_dir:
        dataset_dir = dataset_dir[0]
    settings.dataset_dir = extract_dir_path_from_display(dataset_dir) if dataset_dir else ""
    settings.gpu_type = gpu_type
    settings.gpu_count = gpu_count
    settings.timeout_hours = timeout_hours
    settings.gpu_type_test = gpu_type_test
    settings.gpu_count_test = gpu_count_test
    settings.timeout_hours_test = timeout_hours_test
    settings.enable_ar_bucket = enable_ar_bucket
    settings.resolutions = resolutions
    settings.min_ar = min_ar
    settings.max_ar = max_ar
    settings.num_ar_buckets = num_ar_buckets
    settings.frame_buckets_min = frame_buckets_min
    settings.frame_buckets_max = frame_buckets_max
    settings.num_repeats = num_repeats
    settings.lora_rank = lora_rank
    settings.test_strength = test_strength
    settings.test_enabled = test_enabled
    settings.test_height = test_height
    settings.test_width = test_width
    settings.test_frames = test_frames
    settings.test_steps = test_steps
    settings.test_prompts = test_prompts
    settings.test_epoch_mode = test_epoch_mode
    settings.test_specific_epochs = test_specific_epochs
    settings.test_latest_n_epochs = test_latest_n_epochs
    settings.test_folder = test_folder
    settings.api_endpoint = api_endpoint
    settings.activation_checkpointing = activation_checkpointing
    settings.gradient_clipping = gradient_clipping
    settings.eval_every_n_epochs = eval_every_n_epochs
    settings.save_every_n_epochs = save_every_n_epochs
    settings.pipeline_stages = pipeline_stages
    settings.checkpoint_mode = checkpoint_mode
    settings.checkpoint_every_n_epochs = checkpoint_every_n_epochs
    settings.checkpoint_every_n_minutes = checkpoint_every_n_minutes
    settings.caching_batch_size = caching_batch_size
    settings.steps_per_print = steps_per_print
    settings.transformer_path = transformer_path
    settings.vae_path = vae_path
    settings.llm_path = llm_path
    settings.clip_path = clip_path
    settings.private_check = private_check
    settings.test_outputs_check = test_outputs_check
    settings.upload_endpoint = upload_endpoint
    settings.download_endpoint = download_endpoint
    settings.repo_upload = repo_upload
    return settings

def create_config_interface():
    config = load_config(Config.Files.MODAL)
    settings = ConfigSettings()
    
    # Scan dataset directories on load
    initial_dataset_dirs = scan_dataset_directories()
    initial_dataset_choices = [choice[0] for choice in initial_dataset_dirs]
    initial_dataset_value = initial_dataset_choices[0] if len(initial_dataset_dirs) == 1 else None
    
    # If only one directory is found, update config with that directory
    if len(initial_dataset_dirs) == 1:
        config[Config.Sections.TRAINING]["dataset_dir"] = get_real_path_from_choice(initial_dataset_dirs[0])
        save_config(config, Config.Files.MODAL)

    with gr.Blocks(
        title="Hunyuan Config", 
        css=".tb-dropdown { min-width: 400px !important; margin: 8px 0; } .tb-dropdown .wrap { border: 1px solid #2b3137 !important; } .center-content { text-align: center; margin-top: 20px; }"
    ) as interface:
        gr.Markdown(
            "# Hunyuan Configuration\n"
            "Edit and save your training configurations. After saving, use Modal CLI commands for different operations."
        )
        
        with gr.Tab("Training Settings"):
            with gr.Group():
                gr.Markdown("### Basic Settings")
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=150):
                        resume_training = gr.Checkbox(
                            value=config[Config.Sections.TRAINING]["resume"],
                            label="Resume Training",
                            info="If checked, choose folder to resume",
                            elem_classes="equal-height",
                            interactive=True
                        )
                        test_enabled = gr.Checkbox(
                            value=config[Config.Sections.INFERENCE]["test_enabled"],
                            label="Enable Testing",
                            info="Enable/disable testing LoRA after training",
                            elem_classes="equal-height",
                            interactive=True
                        )
                    
                    with gr.Column(scale=6):
                        resume_folder = gr.Dropdown(
                            choices=[],
                            value=config[Config.Sections.TRAINING].get("resume_folder", ""),
                            label="Resume From Folder",
                            info="Folder with checkpoints (YYYYMMDD_HH-MM-SS)",
                            allow_custom_value=True,
                            elem_classes="equal-height",
                            interactive=True
                        )
                        resume_refresh_btn = gr.Button("🔄 Get Resume Folders", size="md", elem_classes="refresh-btn")
                    
                    with gr.Column(scale=2):
                        dataset_dir = gr.Dropdown(
                            choices=initial_dataset_choices,
                            value=config[Config.Sections.TRAINING]["dataset_dir"] if not initial_dataset_value else initial_dataset_value,
                            label="Dataset Directory",
                            info="Choose the folder containing the scanned images/videos",
                            elem_classes="equal-height",
                            allow_custom_value=True
                        )
                        dataset_scan_btn = gr.Button("🔍 Scan Dataset", size="md", elem_classes="refresh-btn")
                
                resume_refresh_btn.click(fn=update_resume_folders, outputs=[resume_folder])
                dataset_scan_btn.click(fn=update_dataset_dropdown, outputs=[dataset_dir])
                dataset_dir.change(fn=on_dataset_selected, inputs=[dataset_dir], outputs=[])
            
            with gr.Group():
                gr.Markdown("### Training Hyperparameters")
                
                with gr.Row():
                    epochs = gr.Number(
                        value=config[Config.Sections.TRAINING]["epochs"],
                        label="Training Epochs",
                        info="Total epochs",
                        interactive=True
                    )
                    micro_batch_size = gr.Number(
                        value=config[Config.Sections.TRAINING]["micro_batch_size"],
                        label="Micro Batch Size",
                        info="Per GPU",
                        interactive=True
                    )
                    gradient_accum_steps = gr.Number(
                        value=config[Config.Sections.TRAINING]["gradient_accum_steps"],
                        label="Gradient Accumulation Steps",
                        info="Number of steps to accumulate gradients before updating weights",
                        interactive=True
                    )
                
                with gr.Row():
                    warmup_steps = gr.Number(
                        value=config[Config.Sections.TRAINING]["warmup_steps"],
                        label="Warmup Steps",
                        info="Steps to increase LR",
                        interactive=True
                    )
                    learning_rate = gr.Number(
                        value=config[Config.Sections.TRAINING]["learning_rate"],
                        label="Learning Rate",
                        info="Initial learning rate for training",
                        interactive=True
                    )
                    lora_rank = gr.Number(
                        value=config[Config.Sections.TRAINING]["lora_rank"],
                        label="LoRA Rank",
                        info="Rank of LoRA matrices (higher = more capacity)",
                        interactive=True
                    )
            
            with gr.Accordion("Dataset Settings", open=False):
                dataset_config = config[Config.Sections.TRAINING]["dataset"]
                
                with gr.Row():
                    enable_ar_bucket = gr.Checkbox(
                        value=dataset_config["enable_ar_bucket"],
                        label="Enable AR Bucketing",
                        info="Group images by aspect ratio to optimize training",
                        interactive=True
                    )
                    resolutions = gr.Number(
                        value=dataset_config["resolutions"][0],
                        label="Resolution",
                        info="Base resolution for training (width/height)",
                        interactive=True
                    )
                
                with gr.Row():
                    min_ar = gr.Number(
                        value=dataset_config["min_ar"],
                        label="Min Aspect Ratio",
                        info="Minimum width/height ratio",
                        interactive=True
                    )
                    max_ar = gr.Number(
                        value=dataset_config["max_ar"],
                        label="Max Aspect Ratio",
                        info="Maximum width/height ratio",
                        interactive=True
                    )
                    num_ar_buckets = gr.Number(
                        value=dataset_config["num_ar_buckets"],
                        label="Number of AR Buckets",
                        info="Buckets to group images by aspect ratio",
                        interactive=True
                    )
                
                with gr.Row():
                    frame_buckets_min = gr.Number(
                        value=dataset_config["frame_buckets"][0],
                        label="Min Frames",
                        info="Minimum frames per video",
                        interactive=True
                    )
                    frame_buckets_max = gr.Number(
                        value=dataset_config["frame_buckets"][1],
                        label="Max Frames",
                        info="Maximum frames per video",
                        interactive=True
                    )
                    num_repeats = gr.Number(
                        value=dataset_config["num_repeats"],
                        label="Dataset Repeats",
                        info="Number of times to repeat dataset per epoch",
                        interactive=True
                    )
            
            with gr.Accordion("Advanced Settings", open=False):
                training_config = config[Config.Sections.TRAINING]
                
                with gr.Row():
                    weight_decay = gr.Number(
                        value=training_config["weight_decay"],
                        label="Weight Decay",
                        info="L2 regularization parameter",
                        interactive=True
                    )
                    pipeline_stages = gr.Number(
                        value=training_config["pipeline_stages"],
                        label="Pipeline Stages",
                        info="Number of pipeline parallel stages",
                        interactive=True
                    )
                
                with gr.Row():
                    gradient_clipping = gr.Number(
                        value=training_config["gradient_clipping"],
                        label="Gradient Clipping",
                        info="Maximum gradient norm to prevent explosions",
                        interactive=True
                    )
                    activation_checkpointing = gr.Checkbox(
                        value=training_config["activation_checkpointing"],
                        label="Activation Checkpointing",
                        info="Save memory by recomputing activations",
                        interactive=True
                    )
                
                with gr.Row():
                    caching_batch_size = gr.Number(
                        value=training_config["caching_batch_size"],
                        label="Caching Batch Size",
                        info="Batch size used for data caching",
                        interactive=True
                    )
                    steps_per_print = gr.Number(
                        value=training_config["steps_per_print"],
                        label="Steps Per Print",
                        info="How often to print training progress",
                        interactive=True
                    )
                
                with gr.Row():
                    eval_every_n_epochs = gr.Number(
                        value=training_config["eval_every_n_epochs"],
                        label="Evaluation Frequency",
                        interactive=True
                    )
                    save_every_n_epochs = gr.Number(
                        value=training_config["save_every_n_epochs"],
                        label="Save Frequency",
                        interactive=True
                    )
                
                checkpoint_mode = gr.Radio(
                    choices=[CheckpointModes.EPOCHS, CheckpointModes.MINUTES],
                    value=training_config["checkpoint_mode"],
                    label="Checkpoint Mode",
                    interactive=True
                )
                
                with gr.Row():
                    checkpoint_every_n_epochs = gr.Number(
                        value=training_config.get("checkpoint_frequency", 5) if training_config["checkpoint_mode"] == CheckpointModes.EPOCHS else 5,
                        label="Checkpoint Frequency (Epochs)",
                        visible=training_config["checkpoint_mode"] == CheckpointModes.EPOCHS,
                        interactive=True
                    )
                    checkpoint_every_n_minutes = gr.Number(
                        value=training_config.get("checkpoint_frequency", 60) if training_config["checkpoint_mode"] == CheckpointModes.MINUTES else 60,
                        label="Checkpoint Frequency (Minutes)",
                        visible=training_config["checkpoint_mode"] == CheckpointModes.MINUTES,
                        interactive=True
                    )
                
                checkpoint_mode.change(
                    fn=lambda mode: {
                        checkpoint_every_n_epochs: gr.update(visible=mode == CheckpointModes.EPOCHS),
                        checkpoint_every_n_minutes: gr.update(visible=mode == CheckpointModes.MINUTES)
                    },
                    inputs=[checkpoint_mode],
                    outputs=[checkpoint_every_n_epochs, checkpoint_every_n_minutes]
                )
            
            with gr.Accordion("Model Paths", open=False):
                transformer_urls = [
                    "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_720_cfgdistill_bf16.safetensors",
                    "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"
                ]
                vae_urls = [
                    "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_vae_fp32.safetensors",
                    "https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/hunyuan_video_vae_bf16.safetensors"
                ]
                
                transformer_path = gr.Dropdown(
                    choices=transformer_urls,
                    value=transformer_urls[0],
                    label="Transformer Model",
                    info="⚠️ Requires redeploy if changed",
                    interactive=True,
                    allow_custom_value=False
                )
                vae_path = gr.Dropdown(
                    choices=vae_urls,
                    value=vae_urls[0],
                    label="VAE Model",
                    info="⚠️ Requires redeploy if changed",
                    interactive=True,
                    allow_custom_value=False
                )
                llm_path = gr.Textbox(
                    value="https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer/tree/main",
                    label="LLM Model",
                    interactive=False
                )
                clip_path = gr.Textbox(
                    value="https://huggingface.co/openai/clip-vit-large-patch14/tree/main",
                    label="CLIP Model",
                    interactive=False
                )
            
            with gr.Accordion("Hardware Settings", open=False):
                gr.Markdown("⚠️ Requires redeploy to apply changes")
                
                with gr.Row():
                    gpu_type = gr.Dropdown(
                        choices=["H100", "A100-80GB", "A100-40GB"],
                        value=config[Config.Sections.CORE]["gpu_type"],
                        label="GPU Type",
                        interactive=True
                    )
                    gpu_count = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=config[Config.Sections.CORE]["gpu_count"],
                        label="GPU Count",
                        interactive=True
                    )
                    timeout_hours = gr.Slider(
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=config[Config.Sections.CORE]["timeout_hours"],
                        label="Training Timeout (hours)",
                        interactive=True
                    )
                
                with gr.Row():
                    gpu_type_test = gr.Dropdown(
                        choices=["H100", "A100-80GB", "A100-40GB"],
                        value=config[Config.Sections.CORE]["gpu_type_test"],
                        label="Test GPU Type",
                        interactive=True
                    )
                    gpu_count_test = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=config[Config.Sections.CORE]["gpu_count_test"],
                        label="Test GPU Count",
                        interactive=True
                    )
                    timeout_hours_test = gr.Slider(
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=config[Config.Sections.CORE]["timeout_hours_test"],
                        label="Test Timeout (hours)",
                        interactive=True
                    )
            
            with gr.Row():
                start_training_btn = gr.Button("Start Training", variant="primary", scale=1)
            training_status = gr.Textbox(label="Training Status", interactive=False)
        
        with gr.Tab("Test Settings"):
            inference_config = config[Config.Sections.INFERENCE]
            
            with gr.Row(equal_height=True, visible=False):
                api_endpoint = gr.Textbox(
                    value=get_api_endpoint("comfy"),
                    label="API Endpoint",
                    info="ComfyUI API URL",
                    scale=8,
                    interactive=True
                )
                refresh_btn = gr.Button(
                    "🔄 Get API Endpoint",
                    size="md",
                    elem_classes="refresh-btn",
                    min_width=100
                )
            refresh_btn.click(fn=update_api_endpoint, outputs=[api_endpoint])
            
            with gr.Accordion(label="Setting for Manual Testing", open=True):
                with gr.Row(equal_height=True):
                    test_folder = gr.Dropdown(
                        choices=[],
                        value=inference_config.get("test_folder", ""),
                        label="Test Folder",
                        info="Folder with checkpoints (YYYYMMDD_HH-MM-SS)",
                        allow_custom_value=True,
                        scale=8,
                        interactive=True
                    )
                    refresh_folders_btn = gr.Button(
                        "🔄 Get Test Folders",
                        size="md",
                        elem_classes="refresh-btn",
                        min_width=100
                    )
                refresh_folders_btn.click(fn=update_test_folders, outputs=[test_folder])
                
                with gr.Row(equal_height=True):
                    test_epoch_mode = gr.Radio(
                        choices=[TestModes.LATEST, TestModes.SPECIFIC, TestModes.LATEST_N],
                        value=inference_config["test_mode"],
                        label="Test Epoch Mode", 
                        info="Latest, Specific, or Latest N",
                        elem_classes="radio-group equal-height",
                        interactive=True
                    )
                    test_latest_n_epochs = gr.Number(
                        value=inference_config["test_latest_n"],
                        label="Latest N Epochs",
                        visible=inference_config["test_mode"] == TestModes.LATEST_N,
                        elem_classes="equal-height",
                        interactive=True
                    )
                    test_specific_epochs = gr.Textbox(
                        value=",".join(map(str, inference_config["test_epochs"])),
                        label="Specific Epochs",
                        info="Comma-separated (e.g., 1,5,10)",
                        visible=inference_config["test_mode"] == TestModes.SPECIFIC,
                        elem_classes="equal-height",
                        interactive=True
                    )

                test_epoch_mode.change(
                    fn=lambda mode: {
                        test_specific_epochs: gr.update(visible=mode == TestModes.SPECIFIC),
                        test_latest_n_epochs: gr.update(visible=mode == TestModes.LATEST_N)
                    },
                    inputs=[test_epoch_mode],
                    outputs=[test_specific_epochs, test_latest_n_epochs]
                )
            with gr.Accordion(label="Setting for both manual and auto test loRA after training", open=True):
                with gr.Row():
                    test_strength = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=inference_config["test_strength"],
                        label="LoRA Strength",
                        info="0 = no LoRA, 1 = full",
                        interactive=True
                    )
                    test_height = gr.Number(
                        value=inference_config["test_height"],
                        label="Output Height",
                        info="Height in pixels of generated output",
                        interactive=True
                    )
                    test_width = gr.Number(
                        value=inference_config["test_width"],
                        label="Output Width",
                        info="Width in pixels of generated output",
                        interactive=True
                    )
                    test_frames = gr.Number(
                        value=inference_config["test_frames"],
                        label="Number of Frames",
                        info="1 = image, >1 = video",
                        interactive=True
                    )
                    test_steps = gr.Number(
                        value=inference_config["test_steps"],
                        label="Inference Steps",
                        info="Number of denoising steps",
                        interactive=True
                    )
                
                test_prompts = gr.Textbox(
                    value="\n".join(inference_config["test_prompts"]),
                    label="Test Prompts (one per line)",
                    info="One prompt per line",
                    lines=10,
                    placeholder="A beautiful video, best quality",
                    interactive=True
                )
            
            with gr.Row():
                start_test_lora_btn = gr.Button("Start Test LoRA", variant="primary", scale=1)
            test_status = gr.Textbox(label="Test Status", interactive=False)
        
        with gr.Tab("Test Results"):
            with gr.Row(equal_height=True):
                training_folder = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Training Folder",
                    info="Select to view results",
                    allow_custom_value=True,
                    scale=9,
                    interactive=True
                )
                refresh_results_btn = gr.Button("🔄 Get Test Results", size="md", min_width=100)
            
            media_gallery = gr.Gallery(
                label="Test Results",
                columns=[4],
                rows=[4],
                height="auto",
                allow_preview=True,
                preview=True,
            )
            
            training_folder.change(
                fn=update_results,
                inputs=[training_folder],
                outputs=[media_gallery]
            )
            
            refresh_results_btn.click(
                fn=lambda folder: (update_training_folders(), update_results(folder, force_reload=True)),
                inputs=[training_folder],
                outputs=[training_folder, media_gallery]
            )
        
        with gr.Tab("HuggingFace"):
            with gr.Tabs():
                with gr.Tab("Upload"):
                    upload_endpoint = gr.Textbox(
                        visible=False,
                        value=get_api_endpoint("hf_upload")
                    )
                    download_endpoint = gr.Textbox(
                        visible=False,
                        value=get_api_endpoint("hf_download")
                    )
                    
                    with gr.Row():
                        repo_upload = gr.Textbox(
                            label="Repository",
                            placeholder="username/repo-name",
                            value=config[Config.Sections.HF].get("repo_upload", ""),
                            info="Creates repo if not exists",
                            scale=8,
                            interactive=True
                        )
                        private_check = gr.Checkbox(
                            label="Private Repo",
                            value=config[Config.Sections.HF].get("private_repo", True),
                            scale=2,
                            interactive=True,
                            info="set to private repo if it will be created"
                        )
                        test_outputs_check = gr.Checkbox(
                            label="Include Tests",
                            value=config[Config.Sections.HF].get("upload_test_outputs", True),
                            scale=2,
                            interactive=True,
                            info="Include test outputs in upload"
                        )
                    
                    with gr.Row(equal_height=True):
                        training_folder = gr.Dropdown(
                            choices=[],
                            label="Training Folder",
                            value=None,
                            scale=9,
                            interactive=True
                        )
                        refresh_folders = gr.Button("🔄 Refresh", min_width=100)
                    
                    upload_type = gr.Radio(
                        choices=["Latest", "Specific Epoch", "Specific Step", "Entire Folder"],
                        value="Latest",
                        label="Upload Type",
                        interactive=True
                    )
                    
                    upload_target_path = gr.Textbox(
                        label="Target Path",
                        interactive=False,
                        info="Will upload latest epoch and step"
                    )
                    
                    upload_btn = gr.Button("📤 Upload", variant="primary")
                    upload_status = gr.Textbox(label="Status", interactive=False)
                    
                    def update_upload_components(upload_type: str, folder_display: str):
                        if not folder_display:
                            return gr.update()
                        
                        parsed = parse_folder_display(folder_display)
                        base = parsed['base_folder']
                        full_name = parsed['full_name']
                        
                        if upload_type == "Latest":
                            step_match = re.search(r'step(\d+)', folder_display)
                            epoch_match = re.search(r'epochs\[([\d,]+)\]', folder_display)
                            step = f"global_step{step_match.group(1)}" if step_match else ""
                            epoch = f"epoch{epoch_match.group(1).split(',')[-1]}" if epoch_match else ""
                            target = f"{full_name}/{','.join(filter(None, [step, epoch]))}"
                            interactive = False
                        
                        elif upload_type == "Specific Epoch":
                            epoch_match = re.search(r'epochs\[([\d,]+)\]', folder_display)
                            target = f"{full_name}/epoch{epoch_match.group(1)}" if epoch_match else full_name
                            interactive = True
                        
                        elif upload_type == "Specific Step":
                            step_match = re.search(r'step(\d+)', folder_display)
                            steps = re.findall(r'step(\d+)', folder_display)
                            target = f"{full_name}/global_step{','.join(steps)}" if steps else full_name
                            interactive = True
                        
                        else:  # Entire Folder
                            target = full_name
                            interactive = True
                        
                        return gr.update(
                            value=target,
                            visible=True,
                            interactive=interactive,
                            info={
                                "Latest": "Will upload latest epoch and step",
                                "Specific Epoch": "Upload selected epochs, e.g., 20250210_05-19-13/epoch1,2,3",
                                "Specific Step": "Upload selected steps, e.g., 20250210_05-19-13/global_step1,2,3",
                                "Entire Folder": "Upload entire folder, e.g., 20250210_05-19-13"
                            }[upload_type]
                        )
                    
                    training_folder.change(
                        fn=update_upload_components,
                        inputs=[upload_type, training_folder],
                        outputs=[upload_target_path]
                    )
                    
                    upload_type.change(
                        fn=update_upload_components,
                        inputs=[upload_type, training_folder],
                        outputs=[upload_target_path]
                    )
                    
                    refresh_folders.click(
                        fn=lambda: gr.update(
                            choices=[f[1] for f in get_formatted_folders(force_reload=True)],
                            value=get_formatted_folders(force_reload=True)[0][1] if get_formatted_folders(force_reload=True) else None
                        ),
                        outputs=[training_folder]
                    )
                    
                    def handle_upload(repo, folder, upload_type, target, priv, tests, settings):
                        save_result = update_configs(settings)
                        if save_result.startswith("⚠️"):
                            return save_result
                        
                        if not repo or not folder:
                            return "⚠️ Missing repository or folder"
                        
                        folder = next((f[0] for f in get_formatted_folders(force_reload=True) if f[1] == folder), None)
                        if not folder:
                            return "⚠️ Invalid folder"
                        
                        if upload_type != "Latest" and not target:
                            return "⚠️ Missing target path"
                        
                        params = {
                            "repo": repo,
                            "private": priv,
                            "include_tests": tests,
                            "target": target
                        }
                        
                        response = requests.post(upload_endpoint.value, params=params)
                        result = response.json()
                        
                        if result.get("status") == "success":
                            return f"✅ Upload completed\nUploaded files:\n" + "\n".join(f"- {f}" for f in result.get("uploaded_files", []))
                        else:
                            return f"⚠️ Upload failed: {result.get('message', 'Unknown error')}"
                
                with gr.Tab("Download"):
                    with gr.Group():
                        gr.Markdown("### Download Settings")
                        
                        repo_download = gr.Textbox(
                            label="Repository",
                            placeholder="username/repo-name",
                            info="HuggingFace repository ID",
                            scale=9,
                            interactive=True
                        )
                        
                        refresh_repo = gr.Button("🔄 Refresh", min_width=100)
                        
                        repo_folder = gr.Dropdown(
                            choices=[],
                            label="Repository Folders",
                            info="Select folder from repository. Click Refresh to load.",
                            value=None,
                            interactive=True
                        )
                        
                        download_type = gr.Radio(
                            choices=["Latest", "Specific Epoch", "Specific Step", "Entire Folder"],
                            value="Latest",
                            label="Download Type",
                            info="Will download latest epoch and step",
                            interactive=True
                        )
                        
                        download_target_path = gr.Textbox(
                            label="Target Path",
                            visible=False,
                            info="Path to download (e.g. 20250210_05-19-13/epoch50)",
                            interactive=True
                        )
                        
                        force_download = gr.Checkbox(
                            label="Force Redownload",
                            info="Force redownload even if files exist locally",
                            interactive=True
                        )
                        
                        download_btn = gr.Button("📥 Download", variant="primary")
                        download_status = gr.Textbox(label="Status", interactive=False)
                        
                        def update_download_components(download_type: str, folder_display: str):
                            if not folder_display:
                                return gr.update(value="", visible=False)
                            
                            try:
                                base_folder = folder_display.split(' - ')[0].strip()
                                steps = re.search(r'steps\[([\d,]+)\]', folder_display)
                                epochs = re.search(r'epochs\[([\d,]+)\]', folder_display)
                                
                                if download_type == "Latest":
                                    latest_step = steps.group(1).split(',')[-1] if steps else ""
                                    latest_epoch = epochs.group(1).split(',')[-1] if epochs else ""
                                    target = f"{base_folder}/global_step{latest_step},epoch{latest_epoch}"
                                
                                elif download_type == "Specific Epoch":
                                    target = f"{base_folder}/epoch{epochs.group(1)}" if epochs else base_folder
                                
                                elif download_type == "Specific Step":
                                    target = f"{base_folder}/global_step{steps.group(1)}" if steps else base_folder
                                
                                else:  # Entire Folder
                                    target = base_folder
                                
                                return gr.update(value=target, visible=True, info="Manually edit if needed")
                            
                            except Exception as e:
                                logger.error(f"Error generating download target: {str(e)}")
                                return gr.update(value="")
                        
                        def handle_download(repo: str, download_type: str, target: Optional[str], force: bool) -> str:
                            """Handle download based on selected options"""
                            logger.info("Processing download request:")
                            logger.info(f"Repository: {repo}")
                            logger.info(f"Download type: {download_type}")
                            logger.info(f"Target path: {target}")
                            logger.info(f"Force redownload: {force}")
                            
                            if not repo:
                                return "⚠️ Please enter repository ID"
                                
                            if download_type != "Latest" and not target:
                                return "⚠️ Please enter target path"
                            
                            # Validate the target path based on download type
                            if download_type == "Specific Epoch" and target:
                                if not re.match(r'^[^/]+/epoch\d+(?:,\d+)*$', target):
                                    return "⚠️ Invalid format for Specific Epoch. Use format: folder/epoch1 or folder/epoch1,2,3"
                            
                            elif download_type == "Specific Step" and target:
                                if not re.match(r'^[^/]+/global_step\d+(?:,\d+)*$', target):
                                    return "⚠️ Invalid format for Specific Step. Use format: folder/global_step1 or folder/global_step1,2,3"
                            
                            elif download_type == "Entire Folder" and target:
                                if "/" in target:
                                    return "⚠️ Invalid format for Entire Folder. Use format: foldername (without slashes)"
                            
                            final_target = target if download_type != "Latest" else None
                            
                            try:
                                params = {
                                    "repo": repo,
                                    "force": str(force).lower()
                                }
                                if final_target:
                                    params["target"] = final_target
                                
                                response = requests.post(download_endpoint.value, params=params)
                                result = response.json()
                                
                                if result.get("status") == "success":
                                    target_path = result.get("target", "latest")
                                    return f"✅ Successfully downloaded {target_path}"
                                else:
                                    error_msg = result.get("message", "Unknown error")
                                    return f"⚠️ Download failed: {error_msg}"
                            except Exception as e:
                                logger.error(f"⚠️ Error during download: {str(e)}")
                                return f"⚠️ Error: {str(e)}"
                        
                        refresh_repo.click(fn=update_repo_folders, inputs=repo_download, outputs=repo_folder)
                        repo_folder.change(fn=update_download_components, inputs=[download_type, repo_folder], outputs=download_target_path)
                        download_type.change(fn=update_download_components, inputs=[download_type, repo_folder], outputs=download_target_path)
                        download_btn.click(fn=handle_download, inputs=[repo_download, download_type, download_target_path, force_download], outputs=download_status)
        
        with gr.Tab("TensorBoard"):
            with gr.Group():
                gr.Markdown("### TensorBoard Log Viewer")
                
                with gr.Row(equal_height=True):
                    folder_dropdown = gr.Dropdown(
                        label="📁 Training Folders",
                        choices=[],
                        interactive=True,
                        scale=3,
                        filterable=True,
                        elem_classes="tb-dropdown",
                        info="Select training folder to view logs"
                    )
                    log_dropdown = gr.Dropdown(
                        label="📝 Log Files",
                        choices=[],
                        interactive=True,
                        scale=2,
                        visible=False,
                        info="Select specific log file"
                    )
                    refresh_btn = gr.Button(
                        "🔄 Refresh",
                        variant="secondary",
                        scale=1
                    )
                
                tb_status = gr.HTML(
                    label="TensorBoard Status",
                    elem_classes="center-content",
                )

                # TensorBoard logs cache
                tensorboard_folders_cache = {}
                tensorboard_logs_cache = {}
                tensorboard_logs_cache_lock = threading.RLock()  # Changed to RLock

                def download_tensorboard_logs(log_path: str) -> str:
                    """Download TensorBoard log files from Modal Volume to local folder
                    
                    Args:
                        log_path: Path to log file in Modal Volume
                        
                    Returns:
                        str: Local path where logs were downloaded, or None if error
                    """
                    try:
                        # Get log directory structure
                        log_dir = os.path.dirname(log_path)
                        local_log_dir = Path("tensorboard_logs") / log_dir
                        
                        # Create directory if it doesn't exist
                        os.makedirs(local_log_dir, exist_ok=True)
                        
                        # Get all log files in the same directory
                        volume = modal.Volume.from_name(Volumes.TRAINING)
                        try:
                            entries = volume.listdir(log_dir, recursive=False)
                            log_files = [e.path for e in entries if "events.out.tfevents" in e.path]
                            logger.debug(f"Found {len(log_files)} log files to download")
                        except Exception as e:
                            logger.error(f"Cannot list log directory {log_dir}: {str(e)}")
                            log_files = [log_path]  # If error, only download selected file
                        
                        # Download and save each log file (only if it doesn't exist)
                        files_downloaded = 0
                        for log_file in log_files:
                            log_filename = os.path.basename(log_file)
                            local_log_path = local_log_dir / log_filename
                            
                            # Skip if file already exists (to avoid unnecessary downloads)
                            if os.path.exists(local_log_path):
                                logger.debug(f"File {local_log_path} already exists, skipping")
                                continue
                            
                            # Download log file from Modal Volume
                            try:
                                log_content = volume.read_file(log_file)
                                
                                # Save log file to local directory
                                with open(local_log_path, "wb") as f:
                                    f.write(b"".join(log_content))
                                
                                files_downloaded += 1
                                logger.debug(f"Downloaded log file to {local_log_path}")
                            except Exception as e:
                                logger.error(f"Cannot download file {log_file}: {str(e)}")
                        
                        # Verify log files were downloaded successfully
                        downloaded_files = list(Path(local_log_dir).glob("events.out.tfevents*"))
                        logger.debug(f"Downloaded {files_downloaded} new log files. Directory now contains {len(downloaded_files)} log files.")
                            
                        # Convert to absolute path to avoid any path resolution issues
                        absolute_log_dir = os.path.abspath(os.path.normpath(local_log_dir))
                        logger.debug(f"Using absolute log directory path: {absolute_log_dir}")
                        
                        if not os.access(absolute_log_dir, os.R_OK):
                            logger.error(f"Permission denied for log directory: {absolute_log_dir}")
                        
                        # Thêm verify nội dung file
                        if downloaded_files:
                            first_file = downloaded_files[0]
                            if first_file.stat().st_size == 0:
                                logger.error("Downloaded empty log file")
                                return None
                        
                        return absolute_log_dir
                    except Exception as e:
                        logger.error(f"Failed to download logs: {str(e)}", exc_info=True)
                        return None

                def update_tb_folders(force_reload=False):
                    """Update TensorBoard folders dropdown with cached data
                    
                    Args:
                        force_reload: If True, ignore cache and reload from volume
                        
                    Returns:
                        List with updated Gradio components
                    """
                    # Use cached folders if available and not forced to reload
                    if not force_reload and "folders" in tensorboard_folders_cache:
                        folders = tensorboard_folders_cache["folders"]
                        if folders:
                            folder_choices = [(f[0], f[1]) for f in folders]
                            latest_folder = folders[0][1]
                            latest_logs = folders[0][2]
                            log_choices = [os.path.basename(p) for p in latest_logs]
                            return [
                                gr.update(choices=folder_choices, value=latest_folder),
                                gr.update(choices=log_choices, value=log_choices[0], visible=len(log_choices) > 1),
                                latest_logs[0]
                            ]
                            
                    # Get fresh data from volume
                    folders = get_tensorboard_folders()
                    
                    # Cache the result
                    tensorboard_folders_cache["folders"] = folders
                    
                    if not folders:
                        return [
                            gr.update(choices=[], value=None),
                            gr.update(visible=False),
                            None
                        ]
                    folder_choices = [(f[0], f[1]) for f in folders]
                    latest_folder = folders[0][1]
                    latest_logs = folders[0][2]
                    log_choices = [os.path.basename(p) for p in latest_logs]
                    return [
                        gr.update(choices=folder_choices, value=latest_folder),
                        gr.update(choices=log_choices, value=log_choices[0], visible=len(log_choices) > 1),
                        latest_logs[0]
                    ]

                def start_tensorboard(log_path: str) -> str:                    
                    try:
                        # Check if TensorBoard is installed
                        try:
                            subprocess.run(["tensorboard", "--version"], check=True, capture_output=True, text=True)
                        except Exception:
                            return "⚠️ TensorBoard is not installed. Please install with: pip install tensorboard"
                        
                        # Create base tensorboard_logs directory if it doesn't exist
                        base_dir = Path("tensorboard_logs")
                        if not base_dir.exists():
                            base_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"Created directory {base_dir}")
                            
                        # Get folder path from log path
                        log_dir = os.path.dirname(log_path)
                        expected_local_dir = os.path.join("tensorboard_logs", log_dir)
                            
                        # Check cache for this log path
                        with tensorboard_logs_cache_lock:
                            cached_dir = tensorboard_logs_cache.get(log_path)
                            if cached_dir and os.path.exists(cached_dir):
                                logger.debug(f"Using cached directory: {cached_dir}")
                                local_log_dir = cached_dir
                            else:
                                local_log_dir = download_tensorboard_logs(log_path)
                                if local_log_dir and os.path.exists(local_log_dir):
                                    logger.debug(f"Caching new directory: {local_log_dir}")
                                    tensorboard_logs_cache[log_path] = local_log_dir
                                else:
                                    logger.error("Failed to cache directory")
                        
                        if not local_log_dir:
                            return f"⚠️ Cannot download log files. Check logs for details."
                        
                        # Normalize path for proper comparison
                        local_log_dir = os.path.normpath(local_log_dir)
                        expected_local_dir = os.path.normpath(expected_local_dir)
                        logger.debug(f"Normalized log directory paths: {local_log_dir}, {expected_local_dir}")
                        
                        # Ensure local_log_dir is correct (sometimes cache can have invalid path)
                        if local_log_dir != expected_local_dir and os.path.exists(expected_local_dir):
                            local_log_dir = expected_local_dir
                            logger.debug(f"Corrected log directory path to: {local_log_dir}")
                            # Update cache with corrected path
                            with tensorboard_logs_cache_lock:
                                tensorboard_logs_cache[log_path] = local_log_dir
                        
                        # Double check the directory exists
                        if not os.path.exists(local_log_dir):
                            os.makedirs(local_log_dir, exist_ok=True)
                            logger.warning(f"Had to recreate log directory: {local_log_dir}")
                        
                        # Check if log files actually exist (with more detailed logging)
                        log_pattern = "events.out.tfevents*"
                        log_files = list(Path(local_log_dir).glob(log_pattern))
                        logger.debug(f"Found {len(log_files)} log files matching '{log_pattern}' in {local_log_dir}")
                        
                        # If no log files found with glob, try direct directory listing for debugging
                        if not log_files:
                            try:
                                all_files = os.listdir(local_log_dir)
                                logger.debug(f"Directory contents of {local_log_dir}: {all_files}")
                                
                                # Try downloading logs again if directory exists but is empty
                                if not all_files:
                                    logger.warning(f"Directory exists but is empty, downloading logs again")
                                    local_log_dir = download_tensorboard_logs(log_path)
                                    if local_log_dir:
                                        # Update cache with refreshed data
                                        with tensorboard_logs_cache_lock:
                                            tensorboard_logs_cache[log_path] = local_log_dir
                            except Exception as e:
                                logger.error(f"Error listing directory: {str(e)}")
                        
                        # Final check for log files
                        if not log_files:
                            return f"⚠️ No log files found in directory {local_log_dir}. Try refreshing the folder list."
                            
                        logger.info(f"Found {len(log_files)} log files, starting TensorBoard with {local_log_dir}")
                            
                        # Stop previous TensorBoard if running
                        try:
                            subprocess.run(["taskkill", "/f", "/im", "tensorboard.exe"], 
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            logger.info("Stopped previous TensorBoard instance")
                        except Exception:
                            pass  # Ignore if no process running
                        
                        # Start TensorBoard with local path
                        subprocess.Popen([
                            "tensorboard",
                            "--logdir", local_log_dir,
                            "--port", "6006"
                        ])
                        
                        log_name = os.path.basename(log_path)
                        folder_name = os.path.dirname(log_path).split("/")[-1]
                        return f'''
                        <div style="text-align:center; padding: 10px;">
                            <p>✅ TensorBoard started with {len(log_files)} log files</p>
                            <p>📁 Folder: <b>{folder_name}</b> | 📄 Log: <b>{log_name}</b></p>
                            <a href="http://localhost:6006" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin-top: 10px;">
                                📊 Open TensorBoard
                            </a>
                        </div>
                        '''
                    except Exception as e:
                        logger.error(f"Error starting TensorBoard: {str(e)}")
                        return f"⚠️ Error: {str(e)}"

                def handle_refresh():
                    """Handle refresh button click by forcing reload of folder data"""
                    updates = update_tb_folders(force_reload=True)
                    
                    # Check if log path is None
                    if updates[2] is None:
                        return [
                            updates[0],
                            updates[1],
                            "⚠️ No TensorBoard log files found. Please check if your training has generated any logs."
                        ]
                    
                    return [updates[0], updates[1], start_tensorboard(updates[2])]

                def on_folder_change(folder: str):
                    """Handle folder selection change by updating log files dropdown
                    
                    Args:
                        folder: Selected folder name
                        
                    Returns:
                        List with updated log dropdown and selected log path
                    """
                    if not folder:
                        return [gr.update(visible=False), None]
                    
                    # Get folders from cache or reload if needed
                    folders = tensorboard_folders_cache.get("folders", get_tensorboard_folders())
                    selected = next((f for f in folders if f[1] == folder), None)
                    
                    if not selected:
                        return [gr.update(visible=False), None]
                    
                    # Get log file paths and clear any existing cache for these paths
                    log_paths = selected[2]
                    log_choices = [os.path.basename(p) for p in log_paths]
                    
                    if log_paths:
                        # Pre-download the first log file to speed up subsequent operations
                        first_log = log_paths[0]
                        try:
                            # Only pre-download if not already in cache
                            if first_log not in tensorboard_logs_cache:
                                threading.Thread(
                                    target=download_tensorboard_logs,
                                    args=(first_log,),
                                    daemon=True
                                ).start()
                                logger.info(f"Started background download for {first_log}")
                        except Exception as e:
                            logger.error(f"Error in background download: {str(e)}")
                    
                    return [
                        gr.update(choices=log_choices, value=log_choices[0] if log_choices else None, visible=len(log_choices) > 1),
                        log_paths[0] if log_paths else None
                    ]

                # Event handlers
                refresh_btn.click(
                    fn=lambda: [
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "🔄 Loading data..."
                    ],
                    outputs=[folder_dropdown, log_dropdown, tb_status]
                ).then(
                    fn=handle_refresh,
                    outputs=[folder_dropdown, log_dropdown, tb_status]
                )

                folder_dropdown.change(
                    fn=on_folder_change,
                    inputs=folder_dropdown,
                    outputs=[log_dropdown, tb_status]
                ).then(
                    fn=start_tensorboard,
                    inputs=tb_status,
                    outputs=tb_status
                )

                # Event handler for log dropdown change
                def get_full_log_path(log_name, folder):
                    """Get full log path from log name and folder
                    
                    Args:
                        log_name: Selected log file name
                        folder: Selected folder name
                        
                    Returns:
                        str: Full log file path
                    """
                    if not log_name or not folder:
                        return None
                        
                    folders = tensorboard_folders_cache.get("folders", [])
                    selected = next((f for f in folders if f[1] == folder), None)
                    
                    if not selected or not selected[2]:
                        return None
                        
                    # Find matching log path
                    for path in selected[2]:
                        if os.path.basename(path) == log_name:
                            return path
                            
                    # If no match found, return first log path
                    return selected[2][0] if selected[2] else None
                
                log_dropdown.change(
                    fn=get_full_log_path,
                    inputs=[log_dropdown, folder_dropdown],
                    outputs=tb_status
                ).then(
                    fn=start_tensorboard,
                    inputs=tb_status,
                    outputs=tb_status
                )
        
        with gr.Row():
            save_btn = gr.Button("💾 Save All Configurations", variant="primary", size="lg")
            status = gr.Textbox(label="Status", interactive=False, show_label=False)
        
        # Define all components for inputs
        all_components = [
            epochs, micro_batch_size, gradient_accum_steps, warmup_steps, learning_rate,
            weight_decay, resume_training, resume_folder, dataset_dir, gpu_type,
            gpu_count, timeout_hours, gpu_type_test, gpu_count_test, timeout_hours_test,
            enable_ar_bucket, resolutions, min_ar, max_ar, num_ar_buckets,
            frame_buckets_min, frame_buckets_max, num_repeats, lora_rank, test_strength,
            test_enabled, test_height, test_width, test_frames, test_steps, test_prompts,
            test_epoch_mode, test_specific_epochs, test_latest_n_epochs, test_folder,
            api_endpoint, activation_checkpointing, gradient_clipping, eval_every_n_epochs,
            save_every_n_epochs, pipeline_stages, checkpoint_mode, checkpoint_every_n_epochs,
            checkpoint_every_n_minutes, caching_batch_size, steps_per_print, transformer_path,
            vae_path, llm_path, clip_path, private_check, test_outputs_check, upload_endpoint,
            download_endpoint, repo_upload
        ]
        
        # Attach events to buttons
        save_btn.click(
            fn=lambda *args: update_configs(gather_settings(*args)),
            inputs=all_components,
            outputs=status
        )
        start_training_btn.click(
            fn=lambda *args: save_config_and_start_training(gather_settings(*args)),
            inputs=all_components,
            outputs=training_status
        )
        start_test_lora_btn.click(
            fn=lambda *args: save_config_and_start_test_lora(gather_settings(*args)),
            inputs=all_components,
            outputs=test_status,
            queue=True
        )
        upload_btn.click(
            fn=lambda repo, folder, utype, target, priv, tests, *args: handle_upload(repo, folder, utype, target, priv, tests, gather_settings(*args)),
            inputs=[repo_upload, training_folder, upload_type, upload_target_path, private_check, test_outputs_check] + all_components,
            outputs=upload_status
        )
        
        gr.Markdown("### How to use:\n1. Prepare data in same folder as script\n2. Adjust settings in tabs\n3. Save configurations\n4. Start training or testing\n5. Review results")

    return interface

def get_real_path_from_choice(choice_tuple: tuple) -> str:
    """
    Extract just the folder name from a choice tuple (display_name, path)
    """
    if not choice_tuple or len(choice_tuple) < 2:
        return ""
    
    # Get the path from the tuple
    path = choice_tuple[1]
    
    # Extract just the folder name
    if os.path.isabs(path):
        return os.path.basename(path)
    
    return path

if __name__ == "__main__":
    interface = create_config_interface()
    interface.launch(server_name="localhost", server_port=7860, show_error=True, debug=True, prevent_thread_lock=True)