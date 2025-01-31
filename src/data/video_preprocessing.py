import os
from tqdm import tqdm
import multiprocessing
from functools import partial

from src.features import extract_video_features
from src.data.utils import load_file_names, get_str_from_link


def preprocess_videos(input_video_dir, output_csv_dir, output_video_dir=None, num_processes=None, use_parallel=True):
    """
    Process multiple videos in a directory, either for each video
      sequentially or for multiple videos in parallel.

    Args:
        input_video_dir (str): Directory containing input videos
        output_csv_dir (str): Directory to save output CSV files 
                        with pose data
        output_video_dir (str, optional): Directory to save annotated videos,
                         if None, no videos are saved
        use_parallel (bool): Whether to use parallel processing
        num_processes (int, optional): Number of parallel processes to use.
                         If None, uses the number of CPU cores
        
    Returns:
        Folder (here dataset_name from input_video_dir) in output_csv_dir
        (here ="data/interim/video_keypoints") containing a csv 
        of MediaPipe kepoints and one of bonevectors for each video in the dataset.
    """
    os.makedirs(output_csv_dir, exist_ok=True)

    if use_parallel:
        _process_video_parallel(input_video_dir, output_csv_dir, output_video_dir, num_processes)
    else:
        videos = load_file_names(input_video_dir, output_csv_dir)
        for video in tqdm(videos):
            _process_single_video(video, input_video_dir, output_csv_dir, output_video_dir)



def _process_single_video(video, input_video_dir, output_csv_dir, output_video_dir):
    """
    Process a single video file. This function is called by the
      parallel processing pool.

    Args:
        video (str): Name of the video file
        input_video_dir (str): Directory containing input videos
        output_csv_dir (str): Directory to save output CSV files with
                            pose data
        output_video_dir (str, optional): Directory to save annotated 
                            videos, if None, no videos are saved
        
    Returns:
        A csv of MediaPipe kepoints and one of bonevectors for the given video.
    """

    input_video_path = os.path.join(input_video_dir, video)
    output_kps_csv_path = os.path.join(output_csv_dir, f"{os.path.splitext(get_str_from_link(video))[0]}_kps.csv")
    output_bvs_csv_path = os.path.join(output_csv_dir, f"{os.path.splitext(get_str_from_link(video))[0]}_bvs.csv")

    if output_video_dir:
        output_video_path = os.path.join(output_video_dir, f"{os.path.splitext(get_str_from_link(video))[0]}_with_pose.mp4")
    else:
        output_video_path = None
    
    extract_video_features(input_video_path=input_video_path,
                            output_kps_csv_path=output_kps_csv_path,
                            output_bvs_csv_path=output_bvs_csv_path,
                            output_video_path=output_video_path)



def _process_video_parallel(input_video_dir, output_csv_dir, output_video_dir=None, num_processes=None):
    """
    Process multiple videos in parallel, extracting pose data and optionally creating annotated videos.

    Args:
        input_video_dir (str): Directory containing input videos
        output_csv_dir (str): Directory to save output CSV files with
                             pose data
        output_video_dir (str, optional): Directory to save annotated
                             videos, if None, no videos are saved
        num_processes (int, optional): Number of parallel processes
                             to use. If None, uses the number of CPU cores
        
    Returns:
        Folder (here dataset_name from input_video_dir) in output_csv_dir
        (here ="data/interim/video_keypoints") containing a csv 
        of MediaPipe kepoints and one of bonevectors for each video in the dataset.
    """
    videos = load_file_names(input_video_dir, output_csv_dir)
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    pool = multiprocessing.Pool(processes=num_processes)
    
    process_func = partial(_process_single_video, 
                           input_video_dir=input_video_dir, 
                           output_csv_dir=output_csv_dir, 
                           output_video_dir=output_video_dir)
    
    with tqdm(total=len(videos)) as pbar:
        for _ in pool.imap_unordered(process_func, videos):
            pbar.update()
    
    pool.close()
    pool.join()


