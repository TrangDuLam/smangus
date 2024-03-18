import numpy as np
from typing import List
import os

import torch
import torchaudio
import pandas
import pathlib
import argparse

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import ctc_segmentation

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def get_parser():
    
    parser = argparse.ArgumentParser(description="Convert wave files to alignment csv")
    
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        default=None)
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        default=None)
    
    return parser

def get_word_timestamps(audio : torch.Tensor, samplerate : int, model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor, tokenizer : Wav2Vec2CTCTokenizer = tokenizer):
    
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest", sampling_rate = samplerate)
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
    
    # print('audio.shape[0]', audio.shape[0]) Debug usage
    
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    
    return [{"phoneme" : w, "start" : p[0], "end" : p[1]} for w,p in zip(words, segments)]

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    all_audio = pathlib.Path(args.input_dir).glob('**/*.wav*')
    
    for f in all_audio:

        audio, samplerate = torchaudio.load(f)
        word_timestamps = get_word_timestamps(audio[0], samplerate)
        df = pandas.DataFrame(word_timestamps)
        df.to_csv(f"{args.output_dir}/{f.stem}_align.csv", index=False)