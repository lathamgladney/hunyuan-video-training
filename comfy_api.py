# comfy_api.py
"""
ComfyUI API for video generation
"""

import subprocess
import modal
from pathlib import Path
import json
import uuid
import time
import re
import os
from _utils.constants import Config, Paths
from fastapi import Response
from _shared.app import app, output_volume, cache_volume, config_volume, comfy_output_vol
from _utils.build_image import comfy_image
from _config import cfg

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

@app.cls(
    image=comfy_image,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=2700, # 45 minutes, because of the video generation is slow and if we have multiple requests, it will timeout
    gpu=f"{cfg.core.gpu_type_test}:{cfg.core.gpu_count_test}",
    volumes={
        str(Paths.CACHE.base): cache_volume,
        str(Paths.INFERENCE.output): comfy_output_vol,
        str(Paths.TRAINING.output): output_volume,
        str(Paths.TRAINING.config): config_volume
    }
)
class ComfyAPI:
    def __init__(self):
        """Initialize API with workflow configuration"""
        self.workflow_path = f"{Paths.ROOT}/{Config.Files.WORKFLOW}"
        self.base_workflow = json.loads(Path(self.workflow_path).read_text())
        
    @modal.enter()
    def launch_comfy_background(self):
        """Launch ComfyUI server in background"""
        subprocess.run("comfy launch --background", shell=True, check=True)

    def _update_workflow(self, params: dict, client_id: str) -> dict:
        """Update workflow with parameters and client ID"""
        workflow = self.base_workflow.copy()
        
        workflow["3"]["inputs"].update({
            "width": params["width"],
            "height": params["height"],
            "num_frames": params["num_frames"],
            "steps": params["steps"]
        })
        
        workflow["41"]["inputs"].update({
            "lora": params["lora"],
            "strength": params["strength"]
        })
        
        workflow["30"]["inputs"]["prompt"] = params["prompt"]

        output_nodes = [
            node for node in workflow.values()
            if node.get("class_type") in {"SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine"}
        ]
        
        if not output_nodes:
            raise RuntimeError("Workflow missing output node")
        
        epoch_num = 0
        lora_path = params["lora"]
        epoch_match = re.search(r"epoch(\d+)", lora_path)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
        
        timestamp = time.strftime("%Y%m%d_%H-%M-%S")
        output_nodes[0]["inputs"]["filename_prefix"] = f"epoch{epoch_num:03d}_prompt01_{timestamp}_api"
        logger.info(f"Updated workflow with filename_prefix: {output_nodes[0]['inputs']['filename_prefix']}")
        
        return workflow

    @modal.method()
    def generate_video(self, params: dict):
        """Process video generation request"""
        client_id = uuid.uuid4().hex
        temp_workflow_path = f"/tmp/{client_id}.json"
        
        lora_path = params["lora"]
        full_lora_path = os.path.join(Paths.TRAINING.output, lora_path)
        if not os.path.exists(full_lora_path):
            raise FileNotFoundError(f"LoRA file not found: {full_lora_path}")
        
        loras_base_dir = f"{Paths.INFERENCE.base}/ComfyUI/models/loras"
        dest_dir = os.path.join(loras_base_dir, os.path.dirname(lora_path))
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(loras_base_dir, lora_path)
        if not os.path.exists(dest_path):
            os.symlink(full_lora_path, dest_path)
            logger.info(f"Created symlink: {dest_path}")
        elif os.path.realpath(dest_path) != os.path.realpath(full_lora_path):
            os.remove(dest_path)
            os.symlink(full_lora_path, dest_path)
            logger.info(f"Updated symlink: {dest_path}")
        
        params["lora"] = lora_path
        updated_workflow = self._update_workflow(params, client_id)
        
        with open(temp_workflow_path, "w") as f:
            json.dump(updated_workflow, f)
        
        cmd = f"comfy run --workflow {temp_workflow_path} --wait --timeout 2700 --verbose"
        subprocess.run(cmd, shell=True, check=True)
        
        output_nodes = [
            node for node in updated_workflow.values()
            if node.get("class_type") in {"SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine"}
        ]
        
        if not output_nodes:
            raise RuntimeError("Workflow missing output node")
        
        file_prefix = output_nodes[0]["inputs"].get("filename_prefix")
        if not file_prefix:
            raise ValueError("Workflow output node missing filename_prefix")
        
        output_dir = Path(f"{Paths.INFERENCE.output}")
        time.sleep(2)
        
        output_files = [f for f in output_dir.iterdir() if f.name.startswith(file_prefix + "_")]
        
        if not output_files:
            raise FileNotFoundError(f"No output file found with prefix '{file_prefix}'")
        
        video_files = [f for f in output_files if f.suffix in ('.mp4', '.webm')]
        chosen_file = video_files[0] if video_files else output_files[0]
        
        return chosen_file.read_bytes()

    @modal.web_endpoint(method="POST")
    def endpoint(self, request: dict):
        """API endpoint to handle generation requests"""
        try:
            params = {
                "strength": request.get("strength", 0.85),
                "width": request.get("width", 512),
                "height": request.get("height", 320),
                "num_frames": request.get("num_frames", 85),
                "steps": request.get("steps", 30),
                "prompt": request["prompt"],
                "lora": request["lora"]
            }
            video_bytes = self.generate_video.local(params)
            media_type = "video/mp4"
            return Response(content=video_bytes, media_type=media_type)
        except FileNotFoundError as e:
            return Response(content=str(e), status_code=404, media_type="text/plain")
        except Exception as e:
            return Response(content=str(e), status_code=500, media_type="text/plain")