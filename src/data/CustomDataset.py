import os 
import pandas as pd
import shutil
from src.config import get_config
import csv

DATA_CONFIG = get_config('data')
ALL_LABELS_PATH = DATA_CONFIG['dataset_advanced']['all_labels_path']
ALL_FILES_DIR = DATA_CONFIG['dataset_advanced']['all_files_dir']


class CustomDataset:
    """
    A class for managing custom datasets, including label creation and file management.
    """

    def __init__(self, new_dataset_id, split_nr_videos = [0,-1], single_annotator=False, all_labels_path = "", all_files_dir="", file_names=[], export_dir = ""):
        """
        Initialize the CustomDataset with paths for labels and the new dataset.

        Args:
            all_labels_path (str): Path to the CSV file containing all labels.
            new_dataset_name (str): Name of the new dataset to be created.
            file_names List[str]: List of file names to create the dataset with.
        """
        if len(all_labels_path)==0:
            self.all_labels_path = ALL_LABELS_PATH
        else:
            self.all_labels_path = all_labels_path
        
        if len(all_files_dir)==0:
            self.all_files_dir = ALL_FILES_DIR
        else:
            self.all_files_dir = all_files_dir

        if single_annotator:
            self.annotator_ids = ['P0']
        else:
            # The list of full annotator ids 'P0' to 'P20',
            self.annotator_ids = [f'P{nr}' for nr in list(range(0,21))]

        if len(export_dir)==0:
            self.export_path = f"data/raw/video/{new_dataset_id}"
        else:
            self.export_path = f"{export_dir}/{new_dataset_id}"
        self.export_labels_path = os.path.join(self.export_path, "labels.csv")
        self.file_names = file_names

        self.new_dataset_id = new_dataset_id
        self.split_nr_videos = split_nr_videos

        # Run pipeline upon object creation.
        os.makedirs(self.export_path, exist_ok=True)
        self.create()


    def get_custom_labels(self):
        """
        Create a labels file with partial data based on specified criteria.

        Args:
            split_nr_videos (int): Number of video titles to split the dataset at.
            annotator_id (list): List of annotator IDs to include.
            export_path (str, optional): Path to export the resulting DataFrame.

        Returns:
            None
        """
        df = pd.read_csv(self.all_labels_path)
        
        df.rename(columns={'URL':'file_name'}, inplace=True) 
        df["file_name"] = df["file_name"].apply(lambda name: name.split('/')[-1])

        
        if len(self.file_names)==0:
            if self.split_nr_videos[1] == -1:
                elected_names = df["file_name"].drop_duplicates().values[self.split_nr_videos[0]:]

            else:
                selected_names = df["file_name"].drop_duplicates().values[self.split_nr_videos[0]:self.split_nr_videos[1]]
            self.file_names = selected_names
        else: 
            selected_names = self.file_names

        df_files = df[df["file_name"].isin(selected_names)]
        df_annotators = df_files[df_files["Annotator ID"].isin(self.annotator_ids)]
        self.get_labels_to_list(df_annotators)
        
        print(f"- Exported custom labels to {self.export_path}. \n")

         
    def get_labels_to_list(self, dataframe, cols_drop = []):
        """ Gather all labels into a list and add them to a single column 'labels'."""

        # Create the output csv file path
        output_file = self.export_labels_path

        # Iterate through each video file and its corresponding labels
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(['file_name', 'annotator_id','labels'])

            for i in range(len(dataframe)):
                row=dataframe.drop(columns = cols_drop).iloc[i,:]
                list_row=list(row.dropna().values)
                writer.writerow([list_row[0], list_row[1], list_row[2:]])
        

    def copy_mp4_files(self):
        """
        Copy MP4 files listed in a labels file to a new directory.

        Args:
            labels_file_path (str): Path to the CSV file containing labels.
            all_files_dir (str): Directory containing all the original MP4 files.
            new_dir (str, optional): Directory to copy the files to. Defaults to self.export_path.
            col (str, optional): Name of the column in the labels file containing file names.

        Raises:
            ValueError: If a file in the labels is not found in the source directory.
        """

        labels_df = pd.read_csv(self.export_labels_path)
        labelled_file_names = labels_df["file_name"].drop_duplicates().values

        already_copied = [name for name in os.listdir(self.export_path) if name.endswith(".mp4")]
        print(f"There are {len(already_copied)} .mp4 files in the export directory. These will not be copied again.")
        files_names = os.listdir(self.all_files_dir)

        copied_files = []
        skipped_files = []
        not_found = []
        for file in labelled_file_names:
            if file not in files_names:
                print (f"- The file {file} is labelled in {self.all_labels_path} but not found in: {self.all_files_dir}.\n")
                not_found.append(file)
            else:
                if file in already_copied:
                    #print(f"- The {file} is already in the export directory. Skipping to the next one.\n")
                    skipped_files.append(file)
                else:
                    shutil.copy(os.path.join(self.all_files_dir, file), os.path.join(self.export_path, file)) 
                    copied_files.append(file)

        print(f"- Copied {len(copied_files)}/{len(labelled_file_names)} Labelled mp4 files.\n")
        print(f" - Skipped {len(skipped_files)}/{len(labelled_file_names)} mp4 files since they were already there.")
        
        if len(not_found)>0:
            print(f" - The following {len(not_found)} mp4 files were not found in the parent directory. {not_found}")

    def create(self):
        self.get_custom_labels()
        self.copy_mp4_files()

# Example usage:
#dataset = CustomDataset(new_dataset_id="another_funky_one", nr_videos=3, single_annotator=False)
