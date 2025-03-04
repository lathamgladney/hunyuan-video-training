from _utils.config_schema import AppConfig
from _utils.constants import Config
from pathlib import Path
import toml
import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

def load_config() -> AppConfig:
    """Load and validate configuration"""
    try:
        config_path = Path(Config.Files.MODAL)
        if not config_path.exists():
            raise ValueError(f"{Config.Files.MODAL} not found")
        raw_config = toml.load(config_path)
        return AppConfig(raw_config)
    except Exception as e:
        raise RuntimeError(f"Error loading config: {str(e)}")

# Initialize config and logger
cfg = load_config()
