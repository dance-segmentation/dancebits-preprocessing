import argparse
from pathlib import Path
from src import get_config

import os
import csv
import pandas as pd


from src.data_pipeline_train import run_data_pipeline_train
from tests import run_train_pipeline_test


def parse_args():
    parser = argparse.ArgumentParser(description="DanceBits: Choreography Video Segmentation")
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Preprocessing parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing pipeline')
    preprocess_parser.add_argument(
        "--labels-path",
        type=str,
        default="data/raw/video/dataset_X/labels.csv",
        help="Path to the labels CSV file"
    )
    preprocess_parser.add_argument(
        "--files-dir",
        type=str,
        default="data/raw/video/dataset_X",
        help="Directory containing all source files"
    )
    preprocess_parser.add_argument(
        "--dataset-id",
        type=str,
        default="test_dataset_X",
        help="Identifier for the new dataset"
    )
    preprocess_parser.add_argument(
        "--files",
        nargs='+',
        default=["gJS_sFM_c01_d03_mJS2_ch03.mp4", "gHO_sFM_c01_d20_mHO1_ch09.mp4"],
        help="List of files to process"
    )
    
    # Test parser
    test_parser = subparsers.add_parser('test_preprocess', help='Run preprocessing pipeline tests')

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'preprocess':
        # Run preprocessing pipeline
        results = run_data_pipeline_train(
            all_labels_path=args.labels_path,
            all_files_dir=args.files_dir,
            new_dataset_id=args.dataset_id,
            file_names=args.files
        )
        
        # Print results
        print("\nPreprocessing Results:")
        for stage, success in results.items():
            print(f"{stage}: {'✓' if success else '×'}")
            
    elif args.mode == 'test_preprocess':
        print("Running tests for the preprocessing data pipeline for training...")
        run_train_pipeline_test()
    else:
        print("Please specify a mode: 'preprocess' or 'test_preprocess'")


if __name__ == "__main__":
    main()
    

