import pathlib
import os
import wave
import json

from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

def check_sf(input_wav: pathlib.Path):
    """
    Check the sampling frequency of a wav file
    """
    
    fs = 16000
        
    with wave.open(str(input_wav), "rb") as f:
        assert f.getframerate() == fs
        
        
def load_hugginface_model(model_tag: str = 'xlsr-53') :
    '''
    Unified loading interface
    model_tag: str, the tag of the model to be loaded. It is nominated in the config.json file.
    
    '''
    
    json_file = open('./config.json')
    conf = json.load(json_file)
    
    processor = Wav2Vec2Processor.from_pretrained(conf[model_tag])
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(conf[model_tag])
    model = Wav2Vec2ForCTC.from_pretrained(conf[model_tag])
    
    return processor, tokenizer, model

def vad_desilence(filename, hf_token) :
    '''
    Silence removal based on VAD model from pyannote.
    filename: str, the name of the file to be processed.
    hf_token: str, the token for the huggingface model. Due to privacy issues, it is not included in the code.
    '''
    
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=hf_token)
    
    wave, fs = sf.read(filename)
    
    output = pipeline(filename)
    vads = np.array([])
    
    for speech in output.get_timeline().support() :
    
        start, end = speech
    
        t_start = np.floor(fs*start).astype(int)
        t_end = np.floor(fs*end).astype(int)
    
        if len(vads) == 0:
            vads = wave[t_start:t_end+1]
        else :
            vads = np.concatenate((vads, wave[t_start:t_end]))
            
    sf.write(f'{filename[:-4]}_desilenced.wav', vads, fs)