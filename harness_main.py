from atayal_functions import *
from utils import *

import pathlib
import os
import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(description="Core functionality")
    
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Path to input directory",
        default=None)
    
    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        help="Task to perform",
        default=None)
    
    return parser

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()

    files = pathlib.Path(args.input_dir).glob("**/*.wav")
    
    if args.task == "diagnosis":
        
        if not os.path.exists(f"{args.input_dir}_diagnosis"):
            os.makedirs(f"{args.input_dir}_diagnosis")
        
        for f in files:
            diagnosis(f, f"{args.input_dir}_diagnosis")
    
    elif args.task  == "plot_ppg" :
        
        if not os.path.exists(f"{args.input_dir}_ppg_figures"):
            os.makedirs(f"{args.input_dir}_ppg_figures")
            
        for f in files:
            plot_ppg(f, f"{args.input_dir}_ppg_figures")
            
    elif args.task == "recognition" :
        
        # No need to create additional folder
        recognition_and_save(args.input_dir, f"{args.input_dir}_phm_recognition.csv")
        
    elif args.task == "alignment" :
        
        if not os.path.exists(f"{args.input_dir}_alignment"):
            os.makedirs(f"{args.input_dir}_alignment")
        
        for f in files:
            phoneme_alignment(f, f"{args.input_dir}_alignment")
            
    elif args.task == "validate_dir" :
        
        for f in files:
            if check_sf(f) == False:
                raise ValueError(f"{f} does not have the correct sampling frequency (16kHz)")
            
        print("All files have the correct sampling frequency (16kHz)")

