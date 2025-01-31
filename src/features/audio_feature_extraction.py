# External
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

# System
import os
from requests.exceptions import RequestException

# Project
from src.config import get_config


# Get the configuration audio parameters for the librosa from the 
# preprocessing.yaml file

config = get_config('data')
N_MELS = config['audio']['n_mels']
FFT_SIZE = config['audio']['fft_size']
SR = config['audio']['sample_rate']


def extract_audio_features(video_path, audio_path, mel_outpath, fps_outpath, delete_audio=True):
        
    # Extract audio from video
    fps = _extract_audio_from_video(video_path=video_path, audio_path=audio_path)
            
    if fps:
        _generate_mel_spectrogram(audio_path=audio_path, output_path=mel_outpath)
        _get_frames_per_beat(fps=fps, 
                             audio_path=audio_path,
                             frames_per_beat_output_path=fps_outpath)
    
        if delete_audio:
            # Delete the extracted audio files to save disk space
            os.remove(audio_path)
            print(f"Deleted the audio file under {audio_path}")

        print(f"Audio preprocessing for {video_path} completed.")
    else:
        print(f"Skipping audio preprocessinf for {video_path} due to previous errors.")



def _extract_audio_from_video(video_path, audio_path):
    """
    Extracts the audio from a single video file and saves it as a wav file 

    Args:
    input_video_path (str): Path to the input video file
    output_csv_path (str): Path to save the output npy file with pose data
    output_video_path (str, optional): Path to save the annotated video, if None, no video is saved

    Raises:
    FileNotFoundError: If the input video file is not found
    IOError: If there's an error opening the video file
    """
    try:
        print(f"Processing video: {video_path}")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        print(f"Extracted audio from {video_path} and saved it to {audio_path}.")
        return video.fps
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False  # Indicate that processing failed


def _generate_mel_spectrogram(audio_path, output_path):
    """
    Makes a Mel spectrogram of a single wav file and saves it as a numpy file (binary).

    Args:
    input_video_path (str): Path to the input video file
    output_csv_path (str): Path to save the output npy file with pose data
    output_video_path (str, optional): Path to save the annotated video, if None, no video is saved

    Raises:
    FileNotFoundError: If the input audio file is not found
    IOError: If there's an error opening the audio file
    """

    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=FFT_SIZE)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = _normalize_and_scale_spectrogram(S_dB)
    np.save(output_path, S_dB)


def _normalize_and_scale_spectrogram(S_dB):
    """
    Normalizes the input spectrogram to the range [0, 1] and then scales it to the range [-0.5, 0.5].

    Parameters:
    S_dB (numpy.ndarray): Input spectrogram in dB.

    Returns:
    numpy.ndarray: Normalized and scaled spectrogram.
    """
    # Step 1: Normalize the spectrogram to the range [0, 1]
    S_dB_min = S_dB.min()
    S_dB_max = S_dB.max()
    S_dB_normalized = (S_dB - S_dB_min) / (S_dB_max - S_dB_min)

    # Step 2: Scale the normalized values to the range [-0.5, 0.5]
    S_dB_normalized_scaled = S_dB_normalized - 0.5

    return S_dB_normalized_scaled


def _get_frames_per_beat(fps, audio_path, frames_per_beat_output_path):
    """Estimate the number of frames in a beat in this file.
        This number is important for generating the probability
        of segmentation for final labels."""
    
    # Estimate tempo 
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Map tempo to number of frames per beat.
    seconds_per_beat = 60 / tempo[0]
    frames_per_beat = seconds_per_beat * fps
    np.save(frames_per_beat_output_path, frames_per_beat)