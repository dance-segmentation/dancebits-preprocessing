"""
Configuration management for the DanceBits project.
"""

import os
import yaml

def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class Config:
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.main_config = load_config(os.path.join(config_dir, 'main.yaml'))
        self.data_config = load_config(os.path.join(config_dir, 'data', self.main_config['data_config']))
        self.model_config = load_config(os.path.join(config_dir, 'model', self.main_config['model_config']))

    def get_config(self, config_name='main'):
        """Get a specific configuration."""
        if config_name == 'main':
            return self.main_config
        elif config_name == 'data':
            return self.data_config
        elif config_name == 'model':
            return self.model_config
        else:
            raise ValueError(f"Unknown config name: {config_name}")

# Create a global instance of Config
config = Config()

# Function to get configuration, can be imported and used in other modules
def get_config(config_name='main'):
    return config.get_config(config_name)
