"""
Configuration loader utility for the CoT Embeddings project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import torch

def find_project_root() -> Path:
    """
    Find the project root directory by looking for config/config.yaml.
    
    Returns:
        Path to the project root directory
    """
    current = Path(__file__).resolve()
    
    # Go up the directory tree looking for config/config.yaml
    for parent in current.parents:
        config_file = parent / "config" / "config.yaml"
        if config_file.exists():
            return parent
    
    # Fallback: assume we're already in the project root
    cwd = Path.cwd()
    config_file = cwd / "config" / "config.yaml"
    if config_file.exists():
        return cwd
    
    # Last resort: look in the parent of the src directory
    src_parent = Path(__file__).resolve().parents[2]  # Go up from src/utils/config_loader.py
    return src_parent

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file. If None or relative,
                    will search for config/config.yaml from project root.
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if config_path is None:
        config_path = "config/config.yaml"
    
    config_file = Path(config_path)
    
    # If path is relative or doesn't exist, try to find it from project root
    if not config_file.is_absolute() or not config_file.exists():
        project_root = find_project_root()
        if config_path.startswith("config/"):
            # Direct config path
            config_file = project_root / config_path
        else:
            # Assume it's just the filename
            config_file = project_root / "config" / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file} (searched from project root: {find_project_root()})")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Post-process configuration with absolute paths
    config = _post_process_config(config, project_root=find_project_root())
    
    return config

def _post_process_config(config: Dict[str, Any], project_root: Path = None) -> Dict[str, Any]:
    """
    Post-process configuration to handle dynamic values.
    
    Args:
        config: Raw configuration dictionary
        project_root: Project root directory for resolving relative paths
        
    Returns:
        Processed configuration dictionary
    """
    if project_root is None:
        project_root = find_project_root()
    # Handle device configuration
    if 'system' in config:
        if config['system']['device'] == 'cuda' and not torch.cuda.is_available():
            config['system']['device'] = 'cpu'
            print("CUDA not available, falling back to CPU")
    
    # Ensure data directories exist and convert to absolute paths
    for key in ['cache_dir', 'processed_dir']:
        if 'dataset' in config and key in config['dataset']:
            path = Path(config['dataset'][key])
            if not path.is_absolute():
                path = project_root / path
            path.mkdir(parents=True, exist_ok=True)
            config['dataset'][key] = str(path)
    
    # Ensure vector index directory exists and convert to absolute path
    if 'vector_index' in config and 'index_path' in config['vector_index']:
        path = Path(config['vector_index']['index_path'])
        if not path.is_absolute():
            path = project_root / path
        path.mkdir(parents=True, exist_ok=True)
        config['vector_index']['index_path'] = str(path)
    
    return config

def get_model_config(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Get model-specific configuration.
    
    Args:
        config: Full configuration dictionary
        model_type: Type of model ('embedding' or 'generation')
        
    Returns:
        Model-specific configuration
    """
    if model_type not in config:
        raise KeyError(f"Model type '{model_type}' not found in configuration")
    
    return config[model_type]

def get_device(config: Dict[str, Any]) -> str:
    """
    Get the device to use for computation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    return config.get('system', {}).get('device', 'cpu')