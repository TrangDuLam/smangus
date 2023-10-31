import torch
import numpy as np
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import pandas as pd

import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def get_parser():
    
    parser = argparse.ArgumentParser(description="Convert .csv to Praat TextGrid")
    
    parser.add_argument(
        "-i", "--input_wav",
        type=str,
        required=True,
        default=None)
    
    parser.add_argument(
        "-o", "--output_fig",
        type=str,
        required=True,
        default=None)
    
    return parser

def plot_ppg(input_wav, output_fig):
    
    waveform, sample_rate = torchaudio.load(input_wav)
    inputs = processor(waveform[0], return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
            logits = model(inputs.input_values).logits.cpu()[0]
            #logits = model(waveform).logits[0]
            probs = torch.nn.functional.softmax(logits,dim=1)
            
    probs[probs <= 1e-4] = 0
    sorted_probs, indices = torch.sort(probs, descending=True)
    
    all_token_detect = indices[:, :3].numpy().ravel()
    token_detected = np.sort(np.array(list(set(all_token_detect))))
    readable_ppg = torch.zeros(probs.shape[0], len(token_detected))

    for i in range(len(sorted_probs)):
        token_vals = indices[i, :3].numpy().ravel()
        sorter = np.argsort(token_detected)
        token_idx = sorter[np.searchsorted(token_detected, token_vals, sorter=sorter)]
        readable_ppg[i, token_idx] = sorted_probs[i, :3]
        
    df = pd.read_csv('./phoneme_tokens.csv', header=None)
    exact_ph = df.iloc[token_detected, 0].values
    
    sns.heatmap(readable_ppg.numpy().T, cmap="YlGnBu", yticklabels=exact_ph)
    plt.savefig(output_fig)
    
if __name__ == "__main__":
        
    parser = get_parser()
    args = parser.parse_args()
        
    plot_ppg(args.input_wav, args.output_fig)