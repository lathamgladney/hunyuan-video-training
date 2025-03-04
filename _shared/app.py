import modal
from _utils.constants import APP_NAME, Volumes
import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

app = modal.App(name=APP_NAME)

# Setup volumes
output_volume = modal.Volume.from_name(Volumes.TRAINING, create_if_missing=True)
cache_volume = modal.Volume.from_name(Volumes.CACHE, create_if_missing=True)
comfy_output_vol = modal.Volume.from_name(Volumes.COMFY, create_if_missing=True)
config_volume = modal.Volume.from_name(Volumes.CONFIG, create_if_missing=True)
data_volume = modal.Volume.from_name(Volumes.DATA, create_if_missing=True)
# Cache volumes
nv_cache_volume = modal.Volume.from_name("nv-cache", create_if_missing=True)
triton_cache_volume = modal.Volume.from_name("triton-cache", create_if_missing=True)
inductor_cache_volume = modal.Volume.from_name("inductor-cache", create_if_missing=True)
