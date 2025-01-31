import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

class LabelProcessor:
    def __init__(self, labels_df, input_video_dir, plots_dir=""):
        self.labels_df = labels_df

        self.input_video_dir = input_video_dir
        
        self.video_name = ""
        self.nr_frames = 0
        self.plots_dir = plots_dir

        self.frames2beat = 0
        self.all_framestamps = {}
        self.probabilities = []


    def get_labels(self, video_name, frames2beat_path=None):
        """
        Identifies the labels list and expands them into an array with dim=nr_frames.

        Args:
            input_video_dir (str): Directory containing input video files
            video_name (str): Identifier of the video that the labels belong to.

        Returns:
            segment_probabilities (np.array(nr_frames)):
                     Array of segmentation probability from labels
                     with 1. or 0 for frame i in nr_frames.
        """
        print(f"Running to LabelProcessor for file {video_name}.")
        self.video_name = video_name
        
        # 1. Get the number of frames and the timestamp at every frame.
        nr_frames, timestamps = self.get_timestamps(os.path.join(self.input_video_dir, video_name))
        self.nr_frames = nr_frames
        
        if frames2beat_path:
          frames2beat = np.load(frames2beat_path)
          self.frames2beat = frames2beat

        all_framestamps = {}
        if not self.labels_df.empty:
          # 2. Find the row(s) in the labels table that corresponds to the video_name.
          video_label_row_mask = self.labels_df["file_name"] == video_name
          self. video_label_row_mask =  video_label_row_mask
        
          #assert int(np.sum(video_label_row_mask)) >= 1

          annotator_ids = self.labels_df[video_label_row_mask]["annotator_id"].values
          
          print(f"\nLabels for file: {self.video_name}\n")

          # 3. Load the labels using the mask for all annotators.
          for a in range(len(annotator_ids)):
            annotator_id = annotator_ids[a]
            annotator_labels = self.labels_df[video_label_row_mask]["labels"].values[a]
            annotator_labels = str(annotator_labels)

            print(f"{annotator_id}: {annotator_labels}")
            if annotator_labels == "[]":
              time_labels = []      
            else:
              time_labels = [float(value) for value in annotator_labels.replace('[','').replace(']','').split(',')]
          
              # 4. Find the frame index just before the timestamp and expand the labels
              framestamps = np.zeros(nr_frames)
              for t in time_labels:
                labels_ids = self.map_label(t=t*1000, timestamps=timestamps)
                framestamps[labels_ids] = 1.

                assert int(np.sum(framestamps)), len(time_labels)
                all_framestamps[annotator_id]=framestamps
        print("\n")
        self.all_framestamps = all_framestamps
        return all_framestamps

    @staticmethod
    def map_label(t, timestamps):
        zeros = np.zeros_like(timestamps)
        ts = np.ones_like(timestamps)*t
        # Find all frames up to timestamp
        mask = ts > timestamps
        zeros[mask] = 1.0
        # Find the index of the last frame before timestamp
        itemindex = np.where(zeros == 1.0)[0][-1]
        return itemindex

    
    def get_timestamps(self, video_path):
        """
        Extract video-specific information from the video path.

        Args:
            video_path (str): Path to locally-stored video.

        Returns:
            total_frames (int): The total number of frames in the video.
            timestamps (List[float]): The timestamp in milliseconds for each frame.
        """
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get the total number of frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        time_stamps = []
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Get the timestamp in milliseconds
            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            time_stamps.append(timestamp_ms)

        # Release the video capture object
        video.release()

        return total_frames, time_stamps

   
    def get_G(self, frame, framestamp):
      sigma = int(self.frames2beat / 3)
      const = np.sqrt(2 * np.pi * sigma**2)
      prob = np.exp((-1 * np.square(framestamp-frame)) / (2 * sigma**2))
      return prob * 1/const

    
    def get_li(self, frame, framestamps):
      norm_G = self.get_G(frame=0, framestamp=0)
      f_G = self.get_G(frame=frame, framestamp = framestamps)

      return np.sum(f_G, axis=0) * 1/norm_G

    def get_prob_labels(self):
      p = np.zeros(self.nr_frames)
      
      if not self.labels_df.empty:
        annotators = list(self.all_framestamps.keys())
        nr_frames = self.nr_frames
        frame_counts = np.arange(0,nr_frames,1)

        
        for annotator in annotators:
          framestamps = np.where(self.all_framestamps[annotator] == 1.0)[0]

          for frame in range(nr_frames):
            p[frame] += self.get_li(frame = frame, framestamps=framestamps)

        p = p * 1/(len(annotators))

      self.probabilities = p
      return p

    def plot_ground_truth(self, frames2beat=0):
        x_frame_counts = np.arange(self.nr_frames)
        frame_stamps = self.all_framestamps

        annotator_id = list(frame_stamps.keys())
        colors = ['red', 'blue', 'green']

        plt.figure(figsize=(20, 3))

        plt.plot(self.probabilities, label="Ground truth", linewidth=2.0)

        for a in range(len(annotator_id)):
          a_id = annotator_id[a]
          indexes = np.where(frame_stamps[a_id] == 0.0)

          frame_stamps[a_id][indexes] = -1
          plt.scatter(x_frame_counts, frame_stamps[a_id], color=colors[a], label=f"{a_id} stamps")

        if frames2beat!=0:
            frames2beat = self.frames2beat
            beat_window = np.ones(self.nr_frames) * -1
            beat_window[0:frames2beat] = 1.0
            plt.scatter(x_frame_counts, beat_window, label = "1 Beat window in frames", color = "orange")
            plt.legend(loc='lower left')
        else:
          plt.legend()

        
        plt.xlabel("Frame")
        plt.xlim([-10, self.nr_frames+10])
        plt.ylabel("Probability")
        plt.ylim([-0.05,1.05])
        plt.yticks(np.linspace(0,1,3))
        plt.title(f"Video: {self.video_name}")
        
        if len(self.plots_dir) <= 1:
            plots_dir = os.path.join(self.input_video_dir, "plots")
            self.plots_dir = plots_dir
     
        os.makedirs(self.plots_dir, exist_ok=True)
        plt.savefig(os.path.join(self.plots_dir, self.video_name[:-4] + "_gt_plot.png"))
        plt.show()


def preprocess_labels(
      labels_file_name='labels.csv',
      audio_path="data/interim/audio_spectrograms/dataset_advanced_100_P0",
      input_video_dir='data/raw/video/dataset_advanced_100_P012', 
      video_names = ['gHO_sFM_c01_d20_mHO1_ch09.mp4', 'gJS_sFM_c01_d03_mJS2_ch03.mp4'],
      to_plot=False):
    """
        Extracts labels from "labels.csv" for training. The labels 
        corresponding to "video_names" in a dataset "input_video_dir.
        Comes in the pipeline after audio preprocessing as beat information
        is used for extending the labels onto corresponding video frames.   

        Args:
            labels_file_name (str): '<filename>.csv' that stores 
                                      the labels for all videos in a dataset.
            audio_path: Directory of the preprocessed audio data for tempo information. 
            input_video_dir (str): Directory of the raw video dataset.
            video_names (List[str]): '<videoname>.mp4' name of the files from the dataset.

        Returns:
            video_names (List[str]) - see two lines above.
            frame_stamps (Dict[annotator_id]) - Array of +-1 that
                                   indicates segment separation.
            prob_distr (Array) - Probability of each frame to indicate the ending
                                of a segment, computed using paper formula.
        """

    labels_path = os.path.join(input_video_dir, labels_file_name)
    frames2beat_path = [os.path.join(audio_path, video_name[:-4]+'_tempo.npy') for video_name in video_names]
    plots_dir = f"data/interim/labels/dataset_advanced_100/plots"

    df = pd.read_csv(labels_path)
    processor = LabelProcessor(labels_df=df, input_video_dir=input_video_dir, plots_dir=plots_dir)
    
    fs = []
    pds = []
    for i in range(len(video_names)):
        frame_stamps = processor.get_labels(video_name=video_names[i],
                                    frames2beat_path=frames2beat_path[i])
        fs.append(frame_stamps)
        prob_distr = processor.get_prob_labels()
        pds.append(prob_distr)
        
        if to_plot:
          processor.plot_ground_truth()

    return video_names, fs, pds
