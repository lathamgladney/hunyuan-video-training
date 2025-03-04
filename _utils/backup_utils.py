"""
Utilities for backing up training configurations and datasets
"""

import os
import glob
import shutil
from typing import Optional, List, Dict
from pathlib import Path

import logging
from _utils.logging_config import configure_logging

if not logging.getLogger().hasHandlers():
    configure_logging()

logger = logging.getLogger(__name__)

# Constants for backup
IGNORE_PATTERNS = {
    'data': [
        'cache',
        '*.tmp',
        '*.log',
        '*.bak'
    ]
}

CRITICAL_CONFIGS = [
    'hunyuan_video.toml',
    'dataset.toml'
]

SUPPORTING_CONFIGS = [
    'modal.toml',
    'comfy.toml'
]

def should_ignore_file(file_path: str, ignore_patterns: List[str]) -> bool:
    """Check if file should be ignored based on patterns
    
    Args:
        file_path: Path to file
        ignore_patterns: List of patterns to ignore
        
    Returns:
        bool: True if file should be ignored
    """
    from fnmatch import fnmatch
    
    # Convert to Path object for easier handling
    path = Path(file_path)
    
    # Check each pattern
    for pattern in ignore_patterns:
        # Check if pattern matches file name or any parent directory
        if fnmatch(path.name, pattern) or any(fnmatch(part, pattern) for part in path.parts):
            return True
    return False

def get_file_changes(src_folder: str, backup_folder: str, ignore_patterns: List[str] = None) -> Dict[str, List[str]]:
    """Get list of changed, new and deleted files
    
    Args:
        src_folder: Source folder to check
        backup_folder: Backup folder to compare against
        ignore_patterns: List of patterns to ignore
        
    Returns:
        Dict with keys: 'changed', 'new', 'deleted' containing lists of file paths
    """
    changes = {
        'changed': [],
        'new': [],
        'deleted': []
    }
    
    # Handle non-existent backup folder
    if not os.path.exists(backup_folder):
        # All files are new
        for root, _, files in os.walk(src_folder):
            for file in files:
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, src_folder)
                if not ignore_patterns or not should_ignore_file(rel_path, ignore_patterns):
                    changes['new'].append(rel_path)
        return changes
    
    # Get all files in source
    src_files = set()
    for root, _, files in os.walk(src_folder):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(src_file, src_folder)
            if not ignore_patterns or not should_ignore_file(rel_path, ignore_patterns):
                src_files.add(rel_path)
                
                # Check if file exists in backup and has changed
                backup_file = os.path.join(backup_folder, rel_path)
                if os.path.exists(backup_file):
                    src_stat = os.stat(src_file)
                    backup_stat = os.stat(backup_file)
                    if (src_stat.st_size != backup_stat.st_size or 
                        src_stat.st_mtime > backup_stat.st_mtime):
                        changes['changed'].append(rel_path)
                else:
                    changes['new'].append(rel_path)
    
    # Find deleted files
    for root, _, files in os.walk(backup_folder):
        for file in files:
            backup_file = os.path.join(root, file)
            rel_path = os.path.relpath(backup_file, backup_folder)
            if not ignore_patterns or not should_ignore_file(rel_path, ignore_patterns):
                if rel_path not in src_files:
                    changes['deleted'].append(rel_path)
    
    return changes

def has_folder_changed(src_folder: str, backup_folder: str, ignore_patterns: List[str] = None) -> bool:
    """Check if folder contents have changed compared to backup
    
    Args:
        src_folder: Source folder to check
        backup_folder: Backup folder to compare against
        ignore_patterns: List of patterns to ignore
        
    Returns:
        bool: True if changes detected, False otherwise
    """
    changes = get_file_changes(src_folder, backup_folder, ignore_patterns)
    return any(changes.values())

def backup_folder(src_folder: str, backup_dir: str, backup_name: str, is_original: bool = True, 
                 ignore_patterns: List[str] = None, critical_files: List[str] = None) -> Optional[str]:
    """Backup a folder with versioning support
    
    Args:
        src_folder: Source folder to backup
        backup_dir: Directory to store backups
        backup_name: Name for the backup folder
        is_original: Whether this is the original version (no versioning needed)
        ignore_patterns: List of patterns to ignore
        critical_files: List of critical files to always backup
        
    Returns:
        str: Path to created backup folder, or None if failed
    """
    try:
        if is_original:
            # Original folder doesn't need versioning
            dest_folder = os.path.join(backup_dir, f"{backup_name}_orig")
        else:
            # Find latest version for modified folders
            existing_versions = glob.glob(os.path.join(backup_dir, f"{backup_name}_mod_v*"))
            if not existing_versions:
                next_version = 1
            else:
                # Extract version numbers and find max
                version_numbers = [
                    int(os.path.basename(v).split('_v')[-1]) 
                    for v in existing_versions
                ]
                next_version = max(version_numbers) + 1
            
            dest_folder = os.path.join(backup_dir, f"{backup_name}_mod_v{next_version}")
        
        logger.info(f"Backing up {src_folder} to {os.path.basename(dest_folder)}...")
        
        # Custom ignore function
        def ignore_func(dir, files):
            ignored = set()
            for f in files:
                rel_path = os.path.relpath(os.path.join(dir, f), src_folder)
                
                # Always include critical files
                if critical_files and any(rel_path.endswith(cf) for cf in critical_files):
                    continue
                    
                # Check ignore patterns
                if ignore_patterns and should_ignore_file(rel_path, ignore_patterns):
                    ignored.add(f)
            return list(ignored)
            
        shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True, ignore=ignore_func)
        return dest_folder
        
    except Exception as e:
        logger.error(f"Failed to backup folder {src_folder}: {str(e)}")
        return None

def backup_training_files(backup_dir: str, config_src: str, dataset_src: str, dataset_name: str, is_resume: bool = False):
    """Backup training configuration and dataset files
    
    Args:
        backup_dir: Directory to store backups
        config_src: Path to config folder
        dataset_src: Path to dataset folder
        dataset_name: Name of dataset folder
        is_resume: Whether this is a resumed training session
    """
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup config files
        config_backup_base = os.path.join(backup_dir, "config_orig")
        config_has_original = os.path.exists(config_backup_base)
        
        if not config_has_original:
            # First time backup - create original
            backup_folder(config_src, backup_dir, "config", 
                        is_original=True,
                        critical_files=CRITICAL_CONFIGS)
        elif is_resume:
            # Check if critical configs changed
            critical_changes = False
            changes = get_file_changes(config_src, config_backup_base)
            
            for change_type in changes.values():
                if any(f.endswith(tuple(CRITICAL_CONFIGS)) for f in change_type):
                    critical_changes = True
                    break
            
            if critical_changes:
                # Create new version if critical configs changed
                backup_folder(config_src, backup_dir, "config",
                            is_original=False,
                            critical_files=CRITICAL_CONFIGS)
            
        # Backup dataset folder
        dataset_backup_base = os.path.join(backup_dir, f"{dataset_name}_orig")
        dataset_has_original = os.path.exists(dataset_backup_base)
        
        if not dataset_has_original:
            # First time backup - create original
            backup_folder(dataset_src, backup_dir, dataset_name,
                        is_original=True,
                        ignore_patterns=IGNORE_PATTERNS['data'])
        elif is_resume and has_folder_changed(dataset_src, dataset_backup_base, IGNORE_PATTERNS['data']):
            # Dataset changed - create new version
            backup_folder(dataset_src, backup_dir, dataset_name,
                        is_original=False,
                        ignore_patterns=IGNORE_PATTERNS['data'])
            
    except Exception as e:
        logger.error(f"Error backing up training files: {str(e)}")
        raise 