import numpy as np
from typing import List
from .utils import *
from tqdm import tqdm

import torch
import torchaudio

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import typing

import pathlib
import datetime
import os

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

def goodness_of_pronunciation(input_wav: os.PathLike, output_dir: os.PathLike = None, save_to_json: bool = False, backend: str = 'xlsr-53') :
    
    processor, tokenizer, model = load_hugginface_model(backend)
    waveform, sample_rate = torchaudio.load(input_wav)
    
    inputs = processor(waveform[0], return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
        
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
    diagosis_dict = {}

    for i in range(probs.shape[0]):
        
        sorted_probs_frame, ph_idx = torch.sort(probs[i], descending=True)
        gop_per_frame = {} # to store GoP in the candidate frame
        
        if sorted_probs_frame[0] < np.log(0.99):
            
            phone_detected = tokenizer.convert_ids_to_tokens(ph_idx[:3].numpy())
            
            for j in range(3) :
                gop_per_frame[phone_detected[j]] = float(sorted_probs_frame[j].numpy() / sorted_probs_frame[0].numpy())
            
            diagosis_dict[i] = gop_per_frame
            
    if save_to_json:
        with open(f"{output_dir}/{input_wav.stem}_gop.json", "w") as f:
            json.dump(diagosis_dict, f)
            f.close()
        
    return diagosis_dict