# DanceBits-Preprocessing: Feature Engineering for AI-Powered Dance Movement Segmentation

## Overview

DanceBits segments dance videos through a multi-stage pipeline to make choreography learning more accessible and efficient. Developed during the Data Science Retreat bootcamp in 2024, it helps dancers learn by automatically segmenting choreographies and providing real-time feedback. 

This repository contains the data preprocessing pipeline and is structured as follows:
1. Video preprocessing to extract pose keypoints
2. Audio feature extraction
3. Label processing and ground truth generation
4. Feature fusion and dataset creation

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/dancebits-preprocessing.git
cd dancebits

# Create and activate conda environment
conda create --name dancebits-preprocessing python=3.8
conda activate dancebits-preprocessing

# Install dependencies
pip install -r requirements.txt

# Run preprocessing pipeline
python main.py preprocess \
    --labels-path data/raw/video/dataset_X/labels.csv \
    --files-dir data/raw/video/dataset_X \
    --dataset-id your_dataset_name
```

## Project Structure

```
dancebits-preprocessing/
├── config/                # Configuration files
│   ├── main.yaml          # Main configuration
│   └── data/              # Data processing configs
├── data/
│   ├── raw/               # Original videos and labels
│   ├── interim/           # Extracted features
│   └── processed/         # Final training datasets
├── src/
│   ├── data/              # Data processing pipelines
│   └── features/          # Feature extraction code
├── tests/                 # Unit and integration tests
└── main.py                # CLI entry point
```

## Data Pipeline

1. **Video Processing**
   - Extracts pose keypoints using [MediaPipe](https://arxiv.org/abs/1906.08172)
   - Generates bone vectors for movement analysis
   - Outputs: keypoint CSVs and bone vector CSVs

2. **Audio Processing**
   - Extracts mel spectrograms from video audio using [Librosa](https://proceedings.mlr.press/v32/mcfee14.html)
   - Detects tempo and beat information
   - Outputs: .npy files with audio features

3. **Label Processing**
   - Processes manual annotations of movement boundaries
   - Generates probability distributions for segment transitions
   - Outputs: Frame-level segmentation labels

**Tutorials** on the individual steps of the pipeline are provided in `notebooks`.

## Technical Stack

The DanceBits system leverages the following technologies:
- **Python** for core development
- **FastAPI** for API services
- **PyTorch** for deep learning models
- **[MediaPipe](https://arxiv.org/abs/1906.08172)** for pose estimation
- **[Librosa](https://proceedings.mlr.press/v32/mcfee14.html)** for audio processing

## Results

Our implementation of the model based on the research from [1] has demonstrated strong performance in segmenting both basic and advanced choreographies. Testing has shown particularly effective results with structured dance routines, enabling significantly more efficient practice sessions compared to traditional video learning methods. 

For more project motivation, details, and outcomes, please access the blog posts of the team members:
- [Cristina's post on DanceBits](https://cristinamelnic.com/projects/dancebits)
- [Arpad's post on DanceBits](https://dusarpad.com/project_posts/dancebits.html)

The deployment-ready app repository can be found [here](https://github.com/dance-segmentation/dance-bits-api).

## CLI Usage

```bash
# Run preprocessing on a sample dataset
python main.py preprocess

# Run preprocessing tests
python main.py test_preprocess

# Run preprocessing on a new dataset
python main.py preprocess \
    --labels-path data/raw/video/dataset_X/labels.csv \
    --files-dir data/raw/video/dataset_X \
    --dataset-id your_dataset_name
```

## Configuration

The pipeline is configured through YAML files in the `config/` directory:

- `main.yaml`: Global settings and paths
- `data/`: Data processing parameters
- `model/`: Model architecture and training settings

Example configuration:
```yaml
data_config: dataset_config.yaml
model_config: model_v1.yaml
paths:
  raw_data: data/raw/video
  features: data/interim
  processed: data/processed
```

---

Developed at Data Science Retreat - Cohort 39  
Authors: [Paras Mehta](https://github.com/parasmehta), [Cristina Melnic](https://github.com/cristina-v-melnic), and [Arpad Dusa](https://github.com/dusarp)

## References

[1] Endo et al. 2024, ["Automatic Dance Video Segmentation for Understanding Choreography"](https://arxiv.org/abs/2405.19727)  
[2] Tsuchida et al. 2019, ["AIST DANCE VIDEO DATABASE: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing"](https://aistdancedb.ongaaccel.jp)  
[3] Lugaresi et al. 2019, ["MediaPipe: A Framework for Building Perception Pipelines"](https://arxiv.org/abs/1906.08172)  
[4] McFee et al. 2015, ["librosa: Audio and Music Signal Analysis in Python"](https://proceedings.mlr.press/v32/mcfee14.html)  

## Acknowledgments

This project was made possible thanks to:
- Training data and segmentation labels from Endo et al. (2024), based on the [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp)
- Project supervision by [Antonio Rueda-Toicen](https://github.com/andandandand) at Data Science Retreat

