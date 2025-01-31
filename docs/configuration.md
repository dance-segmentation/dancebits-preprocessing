# DanceBits Configuration Guide

This document provides an overview of the configuration system used in the DanceBits project. We use YAML files for configuration, which allows for human-readable and easily modifiable settings.

## Configuration File Structure

Our configuration files are located in the `config/` directory with the following structure:

```
config/
├── main.yaml
├── data/
│   ├── preprocessing.yaml
│   └── augmentation.yaml
├── model/
│   ├── architecture.yaml
│   └── training.yaml
└── evaluation/
    └── metrics.yaml
```

## Configuration Files

### main.yaml

This is the entry point for all configurations. It includes general project settings and references to other configuration files.

Example content:
```yaml
project_name: DanceBits
version: 1.0
data_config: preprocessing.yaml
model_config: architecture.yaml
```

### data/preprocessing.yaml

This file contains settings related to data preprocessing for both video and audio.

Example content:
```yaml
video:
  frame_rate: 30
  resolution: [640, 480]
audio:
  sample_rate: 44100
  n_mels: 128
keypoint_extraction:
  confidence_threshold: 0.5
  model: "movenet_thunder"
```

### model/architecture.yaml

This file defines the architecture of your machine learning model.

Example content:
```yaml
model_type: LSTM
input_size: 
  video: 51  # Number of keypoints * 3 (x, y, confidence)
  audio: 128  # Number of mel spectrogram features
hidden_size: 256
num_layers: 2
dropout: 0.2
```

## Using Configurations in Code

To use these configurations in your Python code, we provide a `config.py` module in the `src/` directory. Here's how to use it:

1. Importing the configuration:

```python
from src.config import get_config

# Get the main configuration
main_config = get_config('main')

# Get the data configuration
data_config = get_config('data')

# Get the model configuration
model_config = get_config('model')
```

2. Accessing configuration values:

```python
# Access video frame rate from data config
frame_rate = data_config['video']['frame_rate']

# Access model type from model config
model_type = model_config['model_type']
```

3. Using configuration in functions:

```python
def preprocess_video(video_path, config=None):
    if config is None:
        config = get_config('data')
    frame_rate = config['video']['frame_rate']
    resolution = config['video']['resolution']
    # Use frame_rate and resolution for preprocessing
```

## Modifying Configurations

To change the behavior of the DanceBits system:

1. Locate the appropriate YAML file in the `config/` directory.
2. Modify the values as needed. Be sure to maintain the YAML structure.
3. Save the file.

Your changes will take effect the next time the configuration is loaded.

## Adding New Configurations

If you need to add new configuration options:

1. Add the new options to the appropriate YAML file.
2. If adding a new configuration file, reference it in `main.yaml`.
3. Update the `Config` class in `src/config.py` if necessary to load the new configuration.

Remember to document any non-obvious configuration settings and their impacts on the system's behavior.

## Best Practices

1. Use descriptive names for configuration keys.
2. Group related configurations together.
3. Use comments in YAML files to explain complex settings.
4. Keep sensitive information (like API keys) out of configuration files. Use environment variables for these.
5. Regularly review and update configurations as your project evolves.

For any questions about the configuration system, please contact the project maintainers.
