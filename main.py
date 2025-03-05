# main.py
"""
Main application to deploy all apps with a single command: `modal deploy main.py`
Includes training, ComfyUI API/UI, and HuggingFace operations.
"""

from _shared.app import app
from training import remote_train
from comfy_api import ComfyAPI
from comfy_ui import comfy_ui
from hf_operations import upload, download
from _utils.logging_config import configure_logging

configure_logging()
__all__ = [remote_train, ComfyAPI, comfy_ui, upload, download]