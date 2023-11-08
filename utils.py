import pathlib
import os
import wave
import json

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

def check_sf(input_wav: pathlib.Path):
    """
    Check the sampling frequency of a wav file
    """
    
    fs = 16000
        
    with wave.open(str(input_wav), "rb") as f:
        assert f.getframerate() == fs
        
        
def load_hugginface_model(model_tag: str = 'xlsr-53') :
    
    json_file = open(f'./config.json')
    conf = json.load(json_file)
    
    processor = Wav2Vec2Processor.from_pretrained(conf[model_tag])
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(conf[model_tag])
    model = Wav2Vec2ForCTC.from_pretrained(conf[model_tag])
    
    return processor, tokenizer, model