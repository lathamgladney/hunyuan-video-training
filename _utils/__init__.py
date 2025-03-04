"""
Utils package for Hunyuan training
"""
import logging

logger = logging.getLogger(__name__)

# Define what should be available in the public API
__all__ = [
    'normalize_config_paths',
    'find_lora_checkpoints',
    'backup_training_files',
    'backup_folder',
    'has_folder_changed'
]

def __getattr__(name):
    """Lazy import mechanism"""
    if name in __all__:
        if name in ['normalize_config_paths']:
            from .config_normalizer import normalize_config_paths
            return normalize_config_paths
        elif name in ['find_lora_checkpoints']:
            from .common_utils import find_lora_checkpoints
            return find_lora_checkpoints
        elif name in ['backup_training_files', 'backup_folder', 'has_folder_changed']:
            from .backup_utils import backup_training_files, backup_folder, has_folder_changed
            if name == 'backup_training_files':
                return backup_training_files
            elif name == 'backup_folder':
                return backup_folder
            else:
                return has_folder_changed
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 