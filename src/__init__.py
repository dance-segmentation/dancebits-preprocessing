"""
DanceBits Preprocessing: Feature Engineering for Choreography Video Segmentation

This package contains the preprocessing modules for the DanceBits project.
"""

# Import main subpackages
from . import data
from . import features
from .config import get_config
from .data_pipeline_train import run_data_pipeline_train

# Define what should be imported with "from src import *"
__all__ = ['data', 'features', 'get_config', 'run_data_pipeline_train']

# You can also define package-level variables or constants
VERSION = '1.1.0'
AUTHOR = 'Paras Mehta, Cristina Melnic, Arpad Dusa'

# You can even include a brief description of the package
__doc__ = """
DanceBits is a project aimed at segmenting choreography using both video and audio data.
It provides tools for preprocessing data by combining features and labels into a ready state for training. 
Main components:
- data: Tools for preprocessing video and audio data
- features: Functions for extracting and combining features from video and audio
"""