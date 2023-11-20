from atayal_retrieval import *
from atayal_feats import *

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
        "-m", "--mode",
        type=str,
        required=True,
        help="Mode of operation",
        default=None)
    
    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        help="Task to perform",
        default=None)
    
    parser.add_argument(
        "-b", "--backend",
        type=str,
        required=False,
        help="Backend model to use",
        default='xlsr-53')
    
    return parser

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()

    files = pathlib.Path(args.input_dir).glob("**/*.wav")
    
    
    if args.mode == 'retrieval' :
        if args.task == "diagnosis":
            
            if not os.path.exists(f"{args.input_dir}_diagnosis"):
                os.makedirs(f"{args.input_dir}_diagnosis")
            
            for f in files:
                diagnosis(f, f"{args.input_dir}_diagnosis", backend=args.backend)
        
        elif args.task  == "plot_ppg" :
            
            if not os.path.exists(f"{args.input_dir}_ppg_figures"):
                os.makedirs(f"{args.input_dir}_ppg_figures")
                
            for f in files:
                plot_ppg(f, f"{args.input_dir}_ppg_figures", backend=args.backend)
                
        elif args.task == "recognition" :
            
            # No need to create additional folder
            recognition_and_save(args.input_dir, f"{args.input_dir}_phm_recognition_{datetime.datetime.now()}.csv")
            
        elif args.task == "alignment" :
            
            if not os.path.exists(f"{args.input_dir}_alignment"):
                os.makedirs(f"{args.input_dir}_alignment")
            
            for f in files:
                phoneme_alignment(f, f"{args.input_dir}_alignment", backend=args.backend)
                
        else:
            raise ValueError(f"{args.task} is not a valid task in retrieval mode")
    
    elif args.mode == 'processor':
            
        if args.task == "validate_dir" :
            
            for f in files:
                if check_sf(f) == False:
                    raise ValueError(f"{f} does not have the correct sampling frequency (16kHz)")
                
            print("All files have the correct sampling frequency (16kHz)")
            
        if args.task == "extract_ppg" :
            
            if not os.path.exists(f"{args.input_dir}_ppg"):
                os.makedirs(f"{args.input_dir}_ppg")
            
            print("Starting to extract PPG")
            
            t_start = datetime.datetime.now()
            for f in files:
                extract_ppg(f, f"{args.input_dir}_ppg", backend=args.backend)
                
            print(f"Time elapsed: {datetime.datetime.now() - t_start}")
            print("PPG extraction completed successfully!")

