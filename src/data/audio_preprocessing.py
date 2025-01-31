import os
from tqdm import tqdm

from src.features import extract_audio_features
from src.data.utils import get_str_from_link, load_file_names


def preprocess_audios(video_folder, audio_folder):
    """
    Process all video files in a folder by extracting audio information 
    and generating mel spectrograms.

    Key input:
        Folder with video files to extract mel spectrograms from.  

    Key output:
        Outputs mel spectrograms and tempo as numpy files in the audio_temp_folder.

    """
    os.makedirs(audio_folder, exist_ok=True)

    videos = load_file_names(input_video_dir=video_folder, output_csv_dir=audio_folder)

    for video_file in tqdm(videos):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
            
            _process_single_audio(video_file=video_file, video_folder=video_folder, audio_folder=audio_folder)
            


def _process_single_audio(video_file, video_folder, audio_folder):
    video_path = os.path.join(video_folder, video_file)
            
    audio_filename = get_str_from_link(video_path)[0:-4]
    audio_path = os.path.join(audio_folder, audio_filename + '.wav')
            
    mel_outpath = os.path.join(audio_folder, audio_filename + '.npy')
    fps_outpath = os.path.join(audio_folder, audio_filename + '_tempo.npy')

    print(f"Processing: {video_path}")

    extract_audio_features(video_path=video_path,
                                   audio_path=audio_path,
                                   mel_outpath=mel_outpath,
                                   fps_outpath=fps_outpath)
