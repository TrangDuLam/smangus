import numpy as np
from typing import List

import torch
import torchaudio

import pandas as pd
import matplotlib.pyplot as plt

import pathlib
import os

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from utils import *

def extract_ppg(input_wav: pathlib.Path, output_dir: str, backend : str = 'xlsr-53') :

    processor, _, model = load_hugginface_model(backend)
    
    waveform, sample_rate = torchaudio.load(input_wav)
    inputs = processor(waveform[0], return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
            logits = model(inputs.input_values).logits.cpu()[0]
            #logits = model(waveform).logits[0]
            probs = torch.nn.functional.softmax(logits,dim=1)
            
    np.save(f"{output_dir}/{input_wav.stem}_ppg.npy", probs.numpy())