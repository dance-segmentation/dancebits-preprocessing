import os

def get_str_from_link(word):
    """Extract the last word from a link. Used for video name selection."""
    return word.split('/')[-1]


def load_file_names(input_video_dir, output_csv_dir):
    """Select names of videos in a input folder that do not exist in the output folder."""
    allfiles = os.listdir(input_video_dir)

    # Check if name from name.mp4 is contained in the names of the output folder.
    already_processed_files = [name[0:-8] for name in os.listdir(output_csv_dir)]
    print(f"Found {len(already_processed_files)} already processed files. These titles will be ignored.")
    
    videos_to_process = [fname for fname in allfiles if fname.endswith('.mp4') and fname[:-4] not in already_processed_files]
    return videos_to_process
