import pathlib
import os
import wave

def check_sf(input_wav: pathlib.Path):
    """
    Check the sampling frequency of a wav file
    """
    
    fs = 16000
        
    with wave.open(str(input_wav), "rb") as f:
        assert f.getframerate() == fs