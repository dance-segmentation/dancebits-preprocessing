import os

from src.data import CustomDataset
from src.data import preprocess_videos
from src.data import preprocess_audios
from src.data import preprocess_labels
from src.data import merge_dataset


def test_custom_dataset(all_labels_path, all_files_dir, new_dataset_id, file_names=None):
    """Test CustomDataset creation and processing"""
    print("Testing CustomDataset creation...")
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
        assert os.path.exists(export_path), "Dataset directory not created"
        assert os.path.exists(os.path.join(export_path, "labels.csv")), "Labels file not created"
        
        # Verify files were copied
        if file_names:
            for file_name in file_names:
                assert os.path.exists(os.path.join(export_path, file_name)), f"File {file_name} not copied"
        
        print("✓ CustomDataset creation successful")
        return True, export_path
        
    except Exception as e:
        print(f"× CustomDataset creation failed: {str(e)}")
        return False, None
    

def test_video_preprocessing(input_video_dir, output_csv_dir, output_video_dir=None):
    """Test video preprocessing pipeline"""
    print("Testing video preprocessing...")
    try:
        # Process videos using the existing function
        preprocess_videos(
            input_video_dir=input_video_dir,
            output_csv_dir=output_csv_dir,
            output_video_dir=output_video_dir,
            use_parallel=False
        )
        
        # Verify outputs exist
        assert os.path.exists(output_csv_dir), "Output directory not created"
        files = os.listdir(output_csv_dir)
        assert any(f.endswith('_kps.csv') for f in files), "No keypoint files generated"
        assert any(f.endswith('_bvs.csv') for f in files), "No bone vector files generated"
        
        print("✓ Video preprocessing successful")
        return True
        
    except Exception as e:
        print(f"× Video preprocessing failed: {str(e)}")
        return False

def test_audio_preprocessing(video_folder, audio_folder):
    """Test audio preprocessing pipeline"""
    print("Testing audio preprocessing...")
    try:
        # Process audio using the existing function
        preprocess_audios(
            video_folder=video_folder,
            audio_folder=audio_folder
        )
        
        # Verify outputs exist
        assert os.path.exists(audio_folder), "Audio output directory not created"
        files = os.listdir(audio_folder)
        assert any(f.endswith('.npy') for f in files), "No mel spectrogram files generated"
        assert any(f.endswith('_tempo.npy') for f in files), "No tempo files generated"
        
        print("✓ Audio preprocessing successful")
        return True
    
    except Exception as e:
        print(f"× Audio preprocessing failed: {str(e)}")
        return False
    
from src.data import preprocess_labels
from src.data import merge_dataset

def test_label_preprocessing(labels_file_name, audio_path, input_video_dir, video_names):
    """Test label preprocessing"""
    print("Testing label preprocessing...")
    try:
        # Process labels using the preprocess_labels function
        video_names, frame_stamps, prob_distr = preprocess_labels(
            labels_file_name=labels_file_name,
            audio_path=audio_path,
            input_video_dir=input_video_dir,
            video_names=video_names,
            to_plot=False
        )
        
        # Verify outputs
        assert len(video_names) > 0, "No videos processed"
        assert len(frame_stamps) > 0, "No frame stamps generated"
        assert len(prob_distr) > 0, "No probability distributions generated"
        
        print("✓ Label preprocessing successful")
        return True
        
    except Exception as e:
        print(f"× Label preprocessing failed: {str(e)}")
        return False
    

def test_dataset_merging(input_video_dir, output_csv_dir, audio_folder, save_dir):
    """Test dataset merging"""
    print("Testing dataset merging...")
    try:
        # Merge dataset using existing function
        merge_dataset(
            input_video_dir=input_video_dir,
            output_csv_dir=output_csv_dir,
            audio_folder=audio_folder,
            save_dir=save_dir
        )
        
        # Verify merged dataset exists
        assert os.path.exists(save_dir), "Merged dataset directory not created"
        
        print("✓ Dataset merging successful")
        return True
        
    except Exception as e:
        print(f"× Dataset merging failed: {str(e)}")
        return False
    
    
def run_train_pipeline_test():
    """Run complete pipeline test"""
    print("\nRunning complete pipeline test...\n")
    

    ALL_LABELS_PATH = "data/raw/video/dataset_X/labels.csv"
    ALL_FILES_DIR = "data/raw/video/dataset_X"
    NEW_DATASET_ID = "test_dataset_X"
    TEST_FILES = [
            "gJS_sFM_c01_d03_mJS2_ch03.mp4",
            "gHO_sFM_c01_d20_mHO1_ch09.mp4"
    ]


    # 1. Test CustomDataset creation
    dataset_ok, dataset_path = test_custom_dataset(
            all_labels_path=ALL_LABELS_PATH,
            all_files_dir=ALL_FILES_DIR,
            new_dataset_id=NEW_DATASET_ID,
            file_names=TEST_FILES
        )
        
    if not dataset_ok:
            print("× Pipeline test stopped due to dataset creation failure")


    # Set up processing directories
    output_csv_dir = f"data/interim/video_keypoints/{NEW_DATASET_ID}"
    audio_folder = f"data/interim/audio_spectrograms/{NEW_DATASET_ID}"
    save_dir = f"data/processed/{NEW_DATASET_ID}"
        

    # 2. Preprocess videos
    print("\n2. Testing Video Preprocessing...")
    video_ok = test_video_preprocessing(
            input_video_dir=dataset_path,
            output_csv_dir=output_csv_dir
        )
        
    # 3. Preprocess audio
    print("\n3. Testing Audio Preprocessing...")
    audio_ok = test_audio_preprocessing(
            video_folder=dataset_path,
            audio_folder=audio_folder
        )
        
    # 4. Process labels
    print("\n4. Testing Label Preprocessing...")
    labels_ok = test_label_preprocessing(
            labels_file_name="labels.csv",
            audio_path=audio_folder,
            input_video_dir=dataset_path,
            video_names=TEST_FILES
        )
    # 5. Merge dataset
    print("\n5. Testing Dataset Merging...")
    if video_ok and audio_ok and labels_ok:
        merge_ok = test_dataset_merging(
                input_video_dir=dataset_path,
                output_csv_dir=output_csv_dir,
                audio_folder=audio_folder,
                save_dir=save_dir
            )
        
    # Report overall status
    print("\nPipeline Test Results:")
    print(f"- Dataset Creation: {'✓' if dataset_ok else '×'}")
    print(f"- Video Preprocessing: {'✓' if video_ok else '×'}")
    print(f"- Audio Preprocessing: {'✓' if audio_ok else '×'}")
    print(f"- Label Preprocessing: {'✓' if labels_ok else '×'}")
    print(f"- Dataset Merging: {'✓' if merge_ok else '×'}")

        
if __name__ == "__main__":
    run_train_pipeline_test()