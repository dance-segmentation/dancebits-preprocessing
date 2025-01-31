import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import cv2
from .LabelProcessor import LabelProcessor
from tqdm import tqdm

def merge_dataset(input_video_dir, output_csv_dir, audio_folder, save_dir, wrap_files=False, format = "pickle", files_to_ignore=[], inference=False):
    """
    Process and merge data for all videos in a dataset.

    This function iterates through all videos in the specified directory,
    retrieving their associated data (keypoints, bone vectors, mel spectrograms, 
    and labels) and creating combined data blocks for each video.

    Args:
    input_video_dir (str): Directory containing input video files
    output_csv_dir (str): Directory containing CSV output files (keypoints, bone vectors, labels)
    audio_folder (str): Directory containing audio feature files (mel spectrograms)
    save_dir (str): Directory to save the processed data blocks
    format (str): ["pickle", "torch", "tensorflow"]

    Steps:
    1. Retrieves file paths for all data types using get_file_paths()
    2. Loads the labels for all videos from a CSV file
    3. For each video:
       - Retrieves specific labels
       - Gets paths for interim files (bone vectors, keypoints, mel spectrogram)
       - Creates a combined data block using create_datablock()

    Note:
    Prints progress messages for each processed video.
    """
    os.makedirs(save_dir + '/' + format, exist_ok=True)

    # 1. Access as lists: 
    # - the video names in the input directory.
    # - the full paths of the visual and audio feature files.
    # - the full path of the labels.csv file in the directory.
    
    # Video id is the name without the .mp4.
    print(f"Ignoring files set for titles: {files_to_ignore}")
    already_merged = [name[:-2] for name in os.listdir(save_dir + "/" + format)]
    print(f"Already merged {len(already_merged)} files will also be ignored.")
    files_to_ignore.extend(already_merged)
    video_names = [file for file in os.listdir(input_video_dir) if file.endswith(".mp4") and file[:-4] not in files_to_ignore]
    
    video_id = [name[0:-4] for name in video_names]
         
    print(len(video_names))

    # 2. Extract the labels table as a pandas dataframe.    
    if inference:
        kps_paths, bvs_paths, mel_spec_paths, frames2beat_paths, labels_path= get_file_paths(input_video_dir, output_csv_dir, audio_folder, inference=inference)
        labels_df = pd.Series()
    else:
        kps_paths, bvs_paths, mel_spec_paths, frames2beat_paths, labels_path = get_file_paths(input_video_dir, output_csv_dir, audio_folder, inference=inference)
        labels_df = pd.read_csv(labels_path[0])
    
    label_processor = LabelProcessor(labels_df=labels_df, input_video_dir=input_video_dir)

    if wrap_files:
        pickle_blocks = {}
        torch_blocks = {}
        tf_blocks = {}

    # 3. Create and save a dictionary with combined features for each video_name.
    for i in tqdm(range(len(video_names))):
        video_name = video_names[i]
        print(f"\nSaving block for {video_name}.")
        
        ## Add f2beat here!
        #frames2beat=frames2beat
        all_interim_file_paths = get_interim_files(video_id[i], kps_paths, bvs_paths, mel_spec_paths,  frames2beat_paths)
        if "" in all_interim_file_paths:
             continue
        else:
            video_bvs_path, video_kps_path, video_spec_path, frames2beat_path = all_interim_file_paths

            framestamps = label_processor.get_labels(video_name=video_name, frames2beat_path=frames2beat_path)
            probabilities = label_processor.get_prob_labels()
            
            frames2beat=label_processor.frames2beat
            nr_frames = label_processor.nr_frames

            # Plot the obtained ground truth labels for videos shown in the paper (figure 7) for a quick test. 
            if video_name in  ['gHO_sFM_c01_d20_mHO1_ch09.mp4', 'gJS_sFM_c01_d03_mJS2_ch03.mp4']:
                label_processor.plot_ground_truth()

            datablock = create_datablock(video_name=video_id[i], 
                            bvs_path=video_bvs_path, 
                            kps_path=video_kps_path,  
                            mel_spec_path=video_spec_path, 
                            labels=probabilities,
                            frames2beat=frames2beat,
                            nr_frames=nr_frames,
                            save_dir=save_dir,
                            format=format)
            
            if datablock !=0:
                print(f"- Created combined features for video: {video_name} in a datablock in directory: {save_dir} \n")

                if wrap_files:
                    # Append datablock in a dictionary of the respective format.
                    if format == "pickle":
                        pickle_blocks[video_name] = convert_to_numpy(datablock)
                    elif format == "torch":
                        torch_blocks[video_name] = convert_to_pytorch(datablock)
                    elif format == "tensorflow": 
                        tf_blocks[video_name] = convert_to_tensorflow(datablock)
            else: 
                 print(f"- Failed to create combined features for video: {video_name} in a datablock in directory: {save_dir} \n")


            
        # Save the joint dictionary for all files in different formats.    
        if wrap_files:
            filedir = save_dir + '/' + "all_datablocks"
            if format == "pickle":
                save_pickle(data_dict=pickle_blocks,
                            filedir=filedir,
                            filename='pickle_blocks.p')
            elif format == "torch":
                save_pytorch_data(data=torch_blocks,
                                filedir=filedir,
                                filename='torch_blocks.pt')
            elif format == "tensorflow": 
                raise ValueError( " Tensorflow format not implemented for wrap=True.")
            
                # TODO: Make separate function for exporting a dict of dicts of numpys. 
                # save_tensorflow_data(data=tf_blocks,
                #                       filedir=filedir,
                #                       filename='tf_blocks.tfrecord')
            
            print(f"- Saved all datablocks from dir: {input_video_dir} into dir: {filedir} \n")


def get_file_paths(input_video_dir, output_csv_dir, audio_folder, inference=False):
    """
    Retrieve file paths for various data types used in video processing.

    This function collects file paths for videos, keypoints, bone vectors,
    mel spectrograms, and labels from specified directories.

    Args:
    input_video_dir (str): Directory containing input video files
    output_csv_dir (str): Directory containing CSV output files (keypoints, bone vectors, labels)
    audio_folder (str): Directory containing audio feature files (mel spectrograms)

    Returns:
    tuple: Contains lists of file paths or names:
        - video_names (list): Names of video files without extension
        - kps_paths (list): Paths to keypoint CSV files
        - bvs_paths (list): Paths to bone vector CSV files
        - mel_spec_paths (list): Paths to mel spectrogram NPY files
        - labels_path (list): Path to the labels CSV file

    Note:
    Uses an internal function `get_format_paths` to retrieve files with specific extensions.
    """

    def get_format_paths(dir_name, end_key = '.mp4', not_key = '_tempo'):
        allfiles = os.listdir(dir_name)
        
        # Get full paths of files.
        files = [ dir_name + '/' + fname for fname in allfiles 
                 if fname.endswith(end_key) and not fname.endswith(not_key+end_key)]
        return files

    #video_names = get_format_paths(dir_name=input_video_dir, end_key='.mp4', only_name=False)
    # kps - keypoints, bvs - bone vectors
    kps_paths = get_format_paths(dir_name=output_csv_dir, end_key='kps.csv')
    bvs_paths = get_format_paths(dir_name=output_csv_dir, end_key='bvs.csv')
    mel_spec_paths = get_format_paths(dir_name=audio_folder, end_key='.npy')
    frames2beat_path = get_format_paths(dir_name=audio_folder, end_key='_tempo.npy')
    if inference:
         return kps_paths, bvs_paths, mel_spec_paths, frames2beat_path, ""
    else:
        labels_path = get_format_paths(dir_name=input_video_dir, end_key='labels.csv')
        return kps_paths, bvs_paths, mel_spec_paths, frames2beat_path, labels_path

    #return video_names, kps_paths, bvs_paths, mel_spec_paths, labels_path
    


def get_interim_files(video_name, kps_paths, bvs_paths, mel_spec_paths, frames2beat_paths):
    """
    Retrieve paths for interim files (bone vectors, keypoints, mel spectrogram) 
    associated with a specific video.

    Args:
    video_name (str): Name of the video
    kps_paths (list): Paths to keypoint files
    bvs_paths (list): Paths to bone vector files
    mel_spec_paths (list): Paths to mel spectrogram files

    Returns:
    tuple: Paths to (bone vectors, keypoints, mel spectrogram) files for the video

    Raises:
    AssertionError: If exactly one file per feature type is not found
    """
    # Get all component interim files for the video by accesing preprocessing files with the same video_name.
    video_bvs_path = [element for element in bvs_paths if video_name in element]
    video_kps_path = [element for element in kps_paths if video_name in element]
    video_spec_path = [element for element in mel_spec_paths if video_name in element]
    frames2beat_path =  [element for element in frames2beat_paths if video_name in element] 

    # There must be exactly one feature file with video_name for each format in the provided directories.    
    assertion_list = [ len(video_bvs_path)== len(video_kps_path),
                       len(video_spec_path) == 1,
                       len(video_kps_path) == len(video_spec_path),
                       len(frames2beat_path)==1]
    if False in assertion_list:
         print(f"Files are missing or incompatible for file: {video_name}.")
         return "", "", "", ""
    else:
        # Return the full path to these feature files.
        return video_bvs_path[0], video_kps_path[0], video_spec_path[0], frames2beat_path[0]


def create_datablock(video_name, bvs_path, kps_path, mel_spec_path, labels, frames2beat, nr_frames, save_dir, format):
    """
    Create and save a data block for a single video, combining multiple feature types.

    This function loads bone vector, keypoint, and mel spectrogram data for a video,
    combines them with labels, and saves the result in multiple formats.

    Args:
    video_name (str): Name of the video file (without extension)
    bvs_path (str): Path to the bone vectors CSV file
    kps_path (str): Path to the keypoints CSV file
    mel_spec_path (str): Path to the mel spectrogram NPY file
    labels (array-like): Labels for the video frames
    save_dir (str): Directory to save the processed data

    Steps:
    1. Loads feature files (bone vectors, keypoints, mel spectrogram)
    2. Combines all data and saves as a pickle file
    3. Converts combined data to TensorFlow format
    4. Saves the TensorFlow data as a TFRecord file

    Note:
    - Pickle data is saved and then immediately loaded for verification
    - PyTorch conversion and saving are currently commented out
    """
        
    # 1. Load the feature files.
    bone_vectors = load_file(file_path=bvs_path, nr_frames=nr_frames, format='.csv')
    keypoints = load_file(file_path=kps_path, nr_frames=nr_frames, format='.csv')
    mel_spec = load_file(file_path=mel_spec_path, nr_frames=0, format='.npy')
    
    assertion_list = [bone_vectors.empty, keypoints.empty, mel_spec.size==0]

    if True in assertion_list:
        print(f"Preprocessing files for {video_name} are missing or are incomplete.")
        
        print(f" Mismatched bone_vecs: {assertion_list[0]} ")
        print(f" Mismatched keypoints: {assertion_list[1]} ")
        print(f" Mismatched mel spectrogram: {assertion_list[2]} ")
        return 0
    else:  
        # 2. Combine all data
        combined_data = combine_data(bone_vectors, keypoints, mel_spec, labels, frames2beat)

        # 3. Export the dict with numpy arrays in a pickle file, pytorch or tf.
        if format == "pickle":
                    save_pickle(convert_to_numpy(combined_data), filedir=f"{save_dir}/pickle", filename=f"{video_name}.p")
        elif format == "torch":
                    pt_data = convert_to_pytorch(combined_data)
                    save_pytorch_data(data=pt_data, filedir=f"{save_dir}/pytorch", filename=f"{video_name}.pt")
        elif format == "tensorflow": 
                    tf_data = convert_to_tensorflow(combined_data)
                    save_tensorflow_data(data=tf_data, filedir=f"{save_dir}/tensorflow", filename=f"{video_name}.tfrecord")
        else:
            raise TypeError("The given format is not implemented.")
                    
        return convert_to_numpy(combined_data)

def load_file(file_path, nr_frames, format = '.csv'):
    """
    Load feature files for a video.

    file_path (str): full path leading to a local file.
                # TODO: implement loading from remote files on gcloud.
    format (str): allowed format of the features files - '.csv' or '.npy'.
    """
    if format == '.csv':
        try:
            file_df = pd.read_csv(file_path)

            file_df=file_df.drop(columns=['frame'])
            if file_df.shape[0] !=nr_frames:
                    print(f"The number of rows and frames do not match in file: {file_path}")
                    return pd.Series()
            else:
                    return file_df

        except pd.errors.EmptyDataError:
            print('Note: filename.csv was empty. Skipping.')
            return pd.Series() # will skip the rest of the block and move to next file
        
    elif format == '.npy':
        file_np = np.load(file_path)
        if file_np.size == 0:
            print(f'File {file_path} is empty!')
            return np.empty()
        else: 
            return file_np
    else:
        raise ValueError("The file format is not supported. Please select either .npy or .csv files.")

def combine_data(bone_vectors, keypoints, mel_spec, labels, frames2beat):
    """
    Combine the data for a file in a single dictionary.

    Args:
        file_name (str)
        bone_vectors (DataFrame (nr_frames*2, 32*2)) - 32 nr joint vectors mediapipe
        keypoints (DataFrame (nr_frames, 33*4)), 33 - nr keypoints mediapipe
        mel_spec (Array())
        labels (Array(nr_frames))
    
    Returns:
        Dictionary with combined args.

    """
    # Bone vectors and keypoints must have dim = nr_frames along axis 0
    assert len(bone_vectors), len(keypoints)
     # Labels and keypoints must have dim = nr_frames along axis 0
    #assert len(keypoints), len(labels)
    assert len(mel_spec), 81
    
    combined = {
        'bone_vectors': bone_vectors,
        'keypoints': keypoints,
        'mel_spectrogram': mel_spec,
        'labels': labels,
        'frames2beat': frames2beat
    }
    return combined
    

import pickle
def save_pickle(data_dict, filedir, filename):
    os.makedirs(filedir, exist_ok = True)
    filepath = filedir + '/' + filename

    with open(filepath, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"- Saved pickle combined data to path: {filepath} \n")

def convert_to_numpy(data):
    # Convert the combined data to numpy arrays
    return {k: np.asarray(v).astype(np.float32) for k, v in data.items()}

def load_pickle(filedir , filename):
    filepath = filedir + '/' + filename
    with open(filepath, 'rb') as fp:
        pickle_data = pickle.load(fp)
        return pickle_data 

def convert_to_pytorch(data):
    # Convert the combined data to PyTorch tensors
    dict_np = convert_to_numpy(data)
    return {k: torch.from_numpy(v) for k, v in dict_np.items()}

def save_pytorch_data(data, filedir, filename):
    os.makedirs(filedir, exist_ok=True)
    filepath = filedir + '/' + filename

    # Save PyTorch data
    torch.save(data, filepath)

    print(f"- Saved pytorch combined data to path: {filepath} \n")

def convert_to_tensorflow(data):
    # Convert the combined data to TensorFlow tensors
    return {k: tf.convert_to_tensor(v.astype(np.float32)) for k, v in data.items()}

def save_tensorflow_data(data, filedir, filename):
    # Save TensorFlow data using TFRecord

    os.makedirs(filedir, exist_ok=True)
    filepath = filedir + '/' + filename

    with tf.io.TFRecordWriter(filepath) as writer:
        feature = {
            k: tf.train.Feature(float_list=tf.train.FloatList(value=v.numpy().flatten()))
            for k, v in data.items()
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    print(f"- Saved tensorflow combined data to path: {filepath} \n")

