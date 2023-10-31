import numpy as np
from typing import List
import argparse

import torch
import torchaudio
import pandas as pd
import pathlib

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def get_parser():
    
    parser = argparse.ArgumentParser(description="Convert .csv to Praat TextGrid")
    
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        default=None)
    
    parser.add_argument(
        "-o", "--output_csv",
        type=str,
        required=False,
        default="atayal_phoneme_result.csv")
    
    return parser

def get_phonemes(audio : np.ndarray, samplerate : int, model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor, tokenizer : Wav2Vec2CTCTokenizer = tokenizer):
    
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest", sampling_rate = samplerate)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        #logits = model(waveform).logits[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    
    return processor.decode(predicted_ids)

def recognition_and_save(input_dir, output_csv):
    
    data_dir = pathlib.Path(input_dir).glob('*.wav')
    df = pd.DataFrame(columns=['file', 'pred_phonemes'])
    
    for file in data_dir:
        waveform, SAMPLERATE = torchaudio.load(file)
        df.loc[len(df)] = [file.name, get_phonemes(waveform[0].numpy(), SAMPLERATE)]
        
    df.to_csv(output_csv, index=False)
    
if __name__ == '__main__':
        
    parser = get_parser()
    args = parser.parse_args()
        
    recognition_and_save(args.input_dir, args.output_csv)