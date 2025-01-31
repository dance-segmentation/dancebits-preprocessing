import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

from src.data import (
    CustomDataset,
    preprocess_videos,
    preprocess_audios,
    preprocess_labels,
    merge_dataset
)


def run_data_pipeline_train(
    all_labels_path: str,
    all_files_dir: str,
    new_dataset_id: str,
    file_names: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        all_labels_path: Path to the labels CSV file
        all_files_dir: Directory containing all source files
        new_dataset_id: Identifier for the new dataset
        file_names: Optional list of specific files to include
        
    Returns:
        Dictionary with status of each pipeline stage
    """
    logger = setup_logger(new_dataset_id)
    logger.info(f"Starting preprocessing pipeline for dataset {new_dataset_id}")
    
    results = {
        'dataset_creation': False,
        'video_processing': False,
        'audio_processing': False,
        'label_processing': False,
        'feature_merging': False
    }
    
    # 1. Create dataset
    dataset_ok, dataset_path = create_dataset(
        all_labels_path=all_labels_path,
        all_files_dir=all_files_dir,
        new_dataset_id=new_dataset_id,
        file_names=file_names
    )
    results['dataset_creation'] = dataset_ok
    
    if not dataset_ok:
        logger.error("Pipeline stopped due to dataset creation failure")
        return results
    
    # Set up processing directories
    output_csv_dir = f"data/interim/video_keypoints/{new_dataset_id}"
    audio_folder = f"data/interim/audio_spectrograms/{new_dataset_id}"
    save_dir = f"data/processed/{new_dataset_id}"
    
    # 2. Process videos
    results['video_processing'] = process_videos(
        input_video_dir=dataset_path,
        output_csv_dir=output_csv_dir
    )
    
    # 3. Process audio
    results['audio_processing'] = process_audio(
        video_folder=dataset_path,
        audio_folder=audio_folder
    )
    
    # 4. Process labels
    results['label_processing'] = process_labels(
        labels_file_name="labels.csv",
        audio_path=audio_folder,
        input_video_dir=dataset_path,
        video_names=file_names if file_names else os.listdir(dataset_path)
    )
    
    # 5. Merge features if all previous steps succeeded
    if all([results['video_processing'], 
            results['audio_processing'], 
            results['label_processing']]):
        results['feature_merging'] = merge_features(
            input_video_dir=dataset_path,
            output_csv_dir=output_csv_dir,
            audio_folder=audio_folder,
            save_dir=save_dir
        )
    
    logger.info("Pipeline complete")
    return results







def setup_logger(dataset_id: str) -> logging.Logger:
    """Set up logging for the preprocessing pipeline."""
    logger = logging.getLogger(f'preprocessing.{dataset_id}')
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    log_dir = Path('logs') / dataset_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    file_handler = logging.FileHandler(log_dir / 'preprocessing.log')
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_dataset(
    all_labels_path: str,
    all_files_dir: str,
    new_dataset_id: str,
    file_names: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Create and organize a new dataset.
    
    Args:
        all_labels_path: Path to the labels CSV file
        all_files_dir: Directory containing all source files
        new_dataset_id: Identifier for the new dataset
        file_names: Optional list of specific files to include
        
    Returns:
        Tuple of (success status, dataset path or None if failed)
    """
    logger = setup_logger(new_dataset_id)
    logger.info(f"Creating dataset {new_dataset_id}")
    
    try:
        # Create CustomDataset instance
        dataset = CustomDataset(
            new_dataset_id=new_dataset_id,
            all_labels_path=all_labels_path,
            all_files_dir=all_files_dir,
            file_names=file_names
        )
        
        # Verify dataset creation
        export_path = f"data/raw/video/{new_dataset_id}"
        if not os.path.exists(export_path):
            raise RuntimeError("Dataset directory not created")
        
        if not os.path.exists(os.path.join(export_path, "labels.csv")):
            raise RuntimeError("Labels file not created")
        
        # Verify files were copied if specific files were requested
        if file_names:
            for file_name in file_names:
                if not os.path.exists(os.path.join(export_path, file_name)):
                    raise RuntimeError(f"File {file_name} not copied")
        
        logger.info("Dataset creation successful")
        return True, export_path
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        return False, None

def process_videos(
    input_video_dir: str,
    output_csv_dir: str,
    output_video_dir: Optional[str] = None
) -> bool:
    """
    Process videos to extract features.
    
    Args:
        input_video_dir: Directory containing input videos
        output_csv_dir: Directory to save output CSV files
        output_video_dir: Optional directory to save annotated videos
        
    Returns:
        Success status
    """
    logger = setup_logger(Path(input_video_dir).parts[-1])
    logger.info("Starting video preprocessing")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_csv_dir, exist_ok=True)
        
        # Process videos
        preprocess_videos(
            input_video_dir=input_video_dir,
            output_csv_dir=output_csv_dir,
            output_video_dir=output_video_dir,
            use_parallel=False
        )
        
        # Verify outputs
        if not os.path.exists(output_csv_dir):
            raise RuntimeError("Output directory not created")
            
        files = os.listdir(output_csv_dir)
        if not any(f.endswith('_kps.csv') for f in files):
            raise RuntimeError("No keypoint files generated")
        if not any(f.endswith('_bvs.csv') for f in files):
            raise RuntimeError("No bone vector files generated")
        
        logger.info("Video preprocessing successful")
        return True
        
    except Exception as e:
        logger.error(f"Video preprocessing failed: {str(e)}")
        return False

def process_audio(
    video_folder: str,
    audio_folder: str
) -> bool:
    """
    Process audio from videos to extract features.
    
    Args:
        video_folder: Directory containing input videos
        audio_folder: Directory to save audio features
        
    Returns:
        Success status
    """
    logger = setup_logger(Path(video_folder).parts[-1])
    logger.info("Starting audio preprocessing")
    
    try:
        # Create output directory
        os.makedirs(audio_folder, exist_ok=True)
        
        # Process audio
        preprocess_audios(
            video_folder=video_folder,
            audio_folder=audio_folder
        )
        
        # Verify outputs
        if not os.path.exists(audio_folder):
            raise RuntimeError("Audio output directory not created")
            
        files = os.listdir(audio_folder)
        if not any(f.endswith('.npy') for f in files):
            raise RuntimeError("No mel spectrogram files generated")
        if not any(f.endswith('_tempo.npy') for f in files):
            raise RuntimeError("No tempo files generated")
        
        logger.info("Audio preprocessing successful")
        return True
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        return False

def process_labels(
    labels_file_name: str,
    audio_path: str,
    input_video_dir: str,
    video_names: List[str]
) -> bool:
    """
    Process labels for the dataset.
    
    Args:
        labels_file_name: Name of the labels file
        audio_path: Path to processed audio features
        input_video_dir: Directory containing input videos
        video_names: List of video files to process
        
    Returns:
        Success status
    """
    logger = setup_logger(Path(input_video_dir).parts[-1])
    logger.info("Starting label preprocessing")
    
    try:
        # Process labels
        video_names, frame_stamps, prob_distr = preprocess_labels(
            labels_file_name=labels_file_name,
            audio_path=audio_path,
            input_video_dir=input_video_dir,
            video_names=video_names,
            to_plot=False
        )
        
        # Verify outputs
        if len(video_names) == 0:
            raise RuntimeError("No videos processed")
        if len(frame_stamps) == 0:
            raise RuntimeError("No frame stamps generated")
        if len(prob_distr) == 0:
            raise RuntimeError("No probability distributions generated")
        
        logger.info("Label preprocessing successful")
        return True
        
    except Exception as e:
        logger.error(f"Label preprocessing failed: {str(e)}")
        return False

def merge_features(
    input_video_dir: str,
    output_csv_dir: str,
    audio_folder: str,
    save_dir: str
) -> bool:
    """
    Merge all extracted features into final dataset.
    
    Args:
        input_video_dir: Directory containing input videos
        output_csv_dir: Directory containing processed video features
        audio_folder: Directory containing processed audio features
        save_dir: Directory to save merged features
        
    Returns:
        Success status
    """
    logger = setup_logger(Path(input_video_dir).parts[-1])
    logger.info("Starting feature merging")
    
    try:
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Merge features
        merge_dataset(
            input_video_dir=input_video_dir,
            output_csv_dir=output_csv_dir,
            audio_folder=audio_folder,
            save_dir=save_dir
        )
        
        # Verify output
        if not os.path.exists(save_dir):
            raise RuntimeError("Merged dataset directory not created")
        
        logger.info("Feature merging successful")
        return True
        
    except Exception as e:
        logger.error(f"Feature merging failed: {str(e)}")
        return False
