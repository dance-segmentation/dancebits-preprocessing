import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional
from src.config import get_config

# Hyperparameters for calling the MediaPipe model.
config = get_config('data')

# Number of predefined MediaPipe keypoints.
nr_mp_keypts = 33
nr_bone_vecs = 35


def extract_video_features(
    input_video_path: str,
    output_kps_csv_path: str,
    output_bvs_csv_path: str,
    output_video_path: Optional[str] = None,
    ) -> None:
    """
    Process a single video, extracting pose data and optionally creating an annotated video.

    Args:
    input_video_path (str): Path to the input video file
    output_kps_csv_path (str): Path to save the output CSV file with pose landmark data
    output_bvs_csv_path (str): Path to save the output CSV file with normalized bone vector data
    output_video_path (str, optional): Path to save the annotated video, if None, no video is saved
    min_detection_confidence (float): Minimum confidence value for the pose detection to be considered successful
    min_tracking_confidence (float): Minimum confidence value for the pose tracking to be considered successful

    Raises:
    FileNotFoundError: If the input video file is not found
    IOError: If there's an error opening the video file
    """
    _validate_input_video(input_video_path)
    _create_output_directories(output_kps_csv_path, output_bvs_csv_path, output_video_path)

    min_detection_confidence = config['video']['min_detection_confidence']
    min_tracking_confidence = config['video']['min_tracking_confidence']
    
    pose = _initialize_mediapipe_pose(min_detection_confidence, min_tracking_confidence)
    video, fps, frame_width, frame_height = _open_video(input_video_path)

    with _create_csv_writers(output_kps_csv_path, output_bvs_csv_path) as (kps_writer, bvs_writer):
        video_writer = _create_video_writer(output_video_path, fps, frame_width, frame_height)
        frame_count = _process_frames(video, pose, kps_writer, bvs_writer, video_writer)

    _cleanup_resources(video, video_writer)
    _print_completion_message(frame_count, output_kps_csv_path, output_bvs_csv_path, output_video_path)

def _validate_input_video(input_video_path: str) -> None:
    """Check if input video exists"""
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

def _create_output_directories(*paths: str) -> None:
    """Create output directories to store bone vectors."""
    for path in paths:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

def _initialize_mediapipe_pose(min_detection_confidence: float, min_tracking_confidence: float) -> mp.solutions.pose.Pose:
    """Create a MediaPipe Pose object for keypoint extraction."""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def _open_video(input_video_path: str) -> Tuple[cv2.VideoCapture, float, int, int]:
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise IOError(f"Error opening video file: {input_video_path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return video, fps, frame_width, frame_height

def _create_csv_writers(output_kps_csv_path: str, output_bvs_csv_path: str):
    class CSVWriters:
        def __init__(self, kps_path: str, bvs_path: str):
            self.kps_file = open(kps_path, 'w', newline='')
            self.bvs_file = open(bvs_path, 'w', newline='')
            self.kps_writer = csv.writer(self.kps_file)
            self.bvs_writer = csv.writer(self.bvs_file)

        def __enter__(self):
            self._write_headers()
            return self.kps_writer, self.bvs_writer

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.kps_file.close()
            self.bvs_file.close()

        def _write_headers(self):
            kps_header = ['frame'] + [f'{coord}_{i}' for i in range(nr_mp_keypts) for coord in ['x', 'y', 'z', 'visibility']]
            bvs_header = ['frame'] + [f'bone_vector_{i}_{coord}' for i in range(nr_bone_vecs) for coord in ['x', 'y']]
            self.kps_writer.writerow(kps_header)
            self.bvs_writer.writerow(bvs_header)

    return CSVWriters(output_kps_csv_path, output_bvs_csv_path)

def _create_video_writer(output_video_path: Optional[str], fps: float, frame_width: int, frame_height: int) -> Optional[cv2.VideoWriter]:
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    return None

def _process_frames(
    video: cv2.VideoCapture,
    pose: mp.solutions.pose.Pose,
    kps_writer: csv.writer,
    bvs_writer: csv.writer,
    video_writer: Optional[cv2.VideoWriter]
) -> int:
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        kps_row, bvs_row = _prepare_data_rows(frame_count, results)
        kps_writer.writerow(kps_row)
        bvs_writer.writerow(bvs_row)

        if video_writer:
            _draw_pose_landmarks(frame, results)
            video_writer.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames.")

    return frame_count

def _prepare_data_rows(frame_count: int, results: List[mp.solutions.pose.PoseLandmark]) -> Tuple[List[float], List[float]]:
    kps_row = [frame_count]
    bvs_row = [frame_count]

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for landmark in landmarks:
            kps_row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        bone_vectors = _compute_bone_vectors(landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        normalized_bone_vectors = _normalize_bone_vectors(bone_vectors)
        for bone_vector in normalized_bone_vectors:
            bvs_row.extend(bone_vector)
    else:
        kps_row.extend([0] * (nr_mp_keypts * 4))
        bvs_row.extend([0] * (nr_bone_vecs * 2))

    return kps_row, bvs_row

def _compute_bone_vectors(landmarks: List[mp.solutions.pose.PoseLandmark], connections: List[Tuple[int, int]]) -> List[np.ndarray]:
    """
    Compute bone vectors from landmarks and connections.

    Args:
    landmarks (list): List of landmarks (keypoints) from MediaPipe Pose.
    connections (list): List of connections between landmarks to create "bones".

    Returns:
    bone_vectors (list): List of bone vectors.
    """
    bone_vectors = []

    for connection in connections:
        start = connection[0]
        end = connection[1]

        start_x, start_y = landmarks[start].x, landmarks[start].y
        end_x, end_y = landmarks[end].x, landmarks[end].y

        bone_vector = np.array([end_x - start_x, end_y - start_y])
        bone_vectors.append(bone_vector)

    return bone_vectors

def _normalize_bone_vectors(bone_vectors: List[np.ndarray], vector_length: float = 0.5) -> List[np.ndarray]:
    """
    Normalize the bone vectors (default length of 0.5 as mentioned in the paper). 

    Args:
    bone_vectors (list): List of bone vectors.
    vector_length (float): Length to normalize the vectors to.

    Returns:
    normalized_bone_vectors (list): List of normalized bone vectors.
    """

    normalized_bone_vectors = []

    for bone_vector in bone_vectors:
        norm = np.linalg.norm(bone_vector)
        normalized_bone_vector = bone_vector / norm if norm != 0 else bone_vector
        normalized_bone_vector *= vector_length
        normalized_bone_vectors.append(normalized_bone_vector)

    return normalized_bone_vectors

def _draw_pose_landmarks(frame: np.ndarray, results: List[mp.solutions.pose.PoseLandmark])-> None:
    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

def _cleanup_resources(video: cv2.VideoCapture, video_writer: Optional[cv2.VideoWriter]) -> None:
    video.release()
    if video_writer:
        video_writer.release()

def _print_completion_message(frame_count: int, output_kps_csv_path: str, output_bvs_csv_path: str, output_video_path: Optional[str]) -> None:
    print(f"Processing complete. Processed {frame_count} frames.")
    print(f"Pose keypoint data saved to {output_kps_csv_path}")
    print(f"Normalized bone vector data saved to {output_bvs_csv_path}")
    if output_video_path:
        print(f"Output video saved to {output_video_path}")