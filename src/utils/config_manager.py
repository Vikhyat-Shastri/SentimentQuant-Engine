"""
Utility module for configuration management.
Loads YAML configuration files and provides easy access to settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class ConfigManager:
    """Manages loading and accessing configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # Load all configuration files
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load all YAML configuration files from config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return
        
        config_files = [
            "sentiment_config.yaml",
            "signal_config.yaml",
            "api_keys.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_name = config_file.replace('.yaml', '')
                        self.configs[config_name] = yaml.safe_load(f)
                    logger.info(f"Loaded configuration: {config_file}")
                except Exception as e:
                    logger.error(f"Error loading {config_file}: {e}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            config_name: Name of config file (without .yaml)
            key_path: Dot-separated path to config value (e.g., 'sentiment.thresholds.fear')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('sentiment_config', 'sentiment.thresholds.fear', 45)
        """
        if config_name not in self.configs:
            logger.warning(f"Configuration '{config_name}' not loaded")
            return default
        
        keys = key_path.split('.')
        value = self.configs[config_name]
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Key '{key_path}' not found in '{config_name}', using default: {default}")
            return default
    
    def get_section(self, config_name: str, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            config_name: Name of config file
            section: Top-level section name
            
        Returns:
            Dictionary of configuration section
        """
        if config_name not in self.configs:
            logger.warning(f"Configuration '{config_name}' not loaded")
            return {}
        
        return self.configs[config_name].get(section, {})
    
    def reload(self) -> None:
        """Reload all configuration files."""
        logger.info("Reloading configurations...")
        self.configs.clear()
        self._load_configs()


# Global configuration instance
config_manager = ConfigManager()
