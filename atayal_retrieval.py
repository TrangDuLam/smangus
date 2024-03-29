import numpy as np
from typing import List
from utils import *
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
import ctc_segmentation

def diagnosis(input_wav: pathlib.Path, output_dir: str, backend : str = 'xlsr-53') :
    
    processor, _, model = load_hugginface_model(backend)
    
    waveform, sample_rate = torchaudio.load(input_wav)
    inputs = processor(waveform[0], return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
            logits = model(inputs.input_values).logits.cpu()[0]
            #logits = model(waveform).logits[0]
            probs = torch.nn.functional.softmax(logits,dim=1)
            
    probs[probs <= 1e-3] = 0
    sorted_probs, indices = torch.sort(probs, descending=True)

    all_token_detect = indices[:, :3].numpy()
    sorted_probs = sorted_probs[:, :3].numpy()
    time_stamps = 0.0125 + 0.02 * np.arange(0, probs.shape[0])

    with open(f"{output_dir}/{input_wav.stem}_phoneme_diagnosis.log", "w") as f:
    
        f.write(f"Datetime : {datetime.datetime.now()}\n")
        f.write(f"Input wav : {input_wav.resolve()}\n")
        f.write("\n")
        f.write("\n")
        
        for i in range(len(all_token_detect)):
        
            if sorted_probs[i][0] <= 0.99 and all_token_detect[i][0] != 0:
                
                f.write(f"[Frame {i}/ Time: {time_stamps[i]} (sec)]\n")
                phonemes = processor.batch_decode(all_token_detect[i])
                phonemes = [p.replace(" ", "") for p in phonemes]
                
                f.write(f"1st detected phoneme {phonemes[0]} with {sorted_probs[i][0]}\n")
                f.write(f"2nd detected phoneme {phonemes[1]} with {sorted_probs[i][1]}\n")
                f.write(f"3rd detected phoneme {phonemes[2]} with {sorted_probs[i][2]}\n")
                f.write("\n")
                
        f.close()
        
        
# need to fix the plot
def plot_log_ppg(input_wav: pathlib.Path, output_dir = str, n_best: int = 3, backend : str = 'xlsr-53'):
    '''
    Visualize the phoneme posteriorgram of the input wav file in the log scale.
    
    '''
    
    processor, tokenizer, model = load_hugginface_model(backend)

    waveform, sample_rate = torchaudio.load(input_wav)
    inputs = processor(waveform[0], return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.log_softmax(logits, dim=-1)

    nbest_ppg = np.log(1e-6) * torch.ones_like(probs)

    for time in range(probs.shape[0]):
        
        if probs[time][0] > torch.log(torch.Tensor([0.999])): # Silence or padding
            nbest_ppg[time][0] = probs[time][0]
        else :
            sort_probs, sort_idx = torch.sort(probs[time],descending=True)
            nbest_ppg[time][sort_idx[:n_best]] = sort_probs[:n_best]

    rawIdxStored = [ i for i, post in enumerate(nbest_ppg.T) if not torch.all(post == np.log(1e-6))]
    # To extract the phonemes that are detected at least once by list comprehension
    
    times = 0.0125 + 0.02 * np.arange(0, nbest_ppg.shape[0]) # the time axis ticks

    plt.figure(figsize=(33.97, 21))
    # sns.set(rc={'figure.figsize':(33.97, 21)})
    sns.set(font_scale=2)
    hmap = sns.heatmap(nbest_ppg[:, rawIdxStored].T, cmap="YlGnBu", cbar_kws={'label': 'log posterior'})
    hmap.set_xticks(np.arange(0, nbest_ppg.shape[0], 25))
    hmap.set_xticklabels(np.round(times[::25], 1), rotation=0)
    hmap.set_yticklabels(tokenizer.convert_ids_to_tokens(rawIdxStored), rotation=0)
    hmap.set_xlabel('Time (s)')
    hmap.set_ylabel('Phonemes')
    hmap.set_title(f'{n_best}-best PPG')
    hmap.figure.savefig(f"{output_dir}/{input_wav.stem}_ppg.png", bbox_inches='tight', dpi=300) # Remove the white space around the figure
    
def get_phonemes(audio : np.ndarray, samplerate : int, backend : str = 'xlsr-53'):
    '''
    Pure phoneme recognition function.
    
    '''
    
    processor, tokenizer, model = load_hugginface_model(backend)
    
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
    
    for file in tqdm(data_dir):
        waveform, SAMPLERATE = torchaudio.load(file)
        df.loc[len(df)] = [file.name, get_phonemes(waveform[0].numpy(), SAMPLERATE)]
        
    df.to_csv(output_csv, index=False)
    
def phoneme_alignment(input_wav: pathlib.Path, output_dir: str, backend : str = 'xlsr-53'):
    '''
    Implement the phoneme alignment algorithm by CTC segmentation.
    
    bug fixing ! Please you atanayl_align_to_phone.py for the reference.
    '''
    
    processor, tokenizer, model = load_hugginface_model(backend)
    
    waveform, sample_rate = torchaudio.load(input_wav)
    
    waveform = waveform[0]
    
    # Run prediction, get logits and probabilities
    inputs = processor(waveform, return_tensors="pt", padding="longest", sampling_rate = sample_rate)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        #logits = model(waveform).logits[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)
    
    # Split the transcription into words
    words = pred_transcript.split(" ")
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    
    # print('waveform.shape[0] : ', waveform.shape[0]) Debug usage
    config.index_duration = waveform.shape[0] / probs.size()[0] / sample_rate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    
    time_stamps = [{"phoneme" : w, "start" : p[0], "end" : p[1]} for w,p in zip(words, segments)]
    
    df = pd.DataFrame(time_stamps)
    df.to_csv(f"{output_dir}/{input_wav.stem}_align.csv", index=False)
    
