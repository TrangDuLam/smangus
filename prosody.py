import pyworld as pw
import numpy as np
import wave
import matplotlib.pyplot as plt
import soundfile as sf

import typing
import numpy.typing as npt

def pitch_contour_pruned(filename: str, to_prune : bool = False) :
    '''
    Extract pitch contour from a wav file using WORLD Vocoder.
    filename: str, the name of the wav file.
    to_prune: bool, determine whether to remove the voiceless frames.
    '''
    
    y, fs = sf.read(filename)
    
    assert fs == 16000, "Sampling rate must be 16000"
    assert len(y.shape) == 1, "Only mono audio files are supported"
    
    f0, _, _ = pw.wav2world(y, fs)
    
    if to_prune:
        f0 = f0[f0 != 0]
    
    return f0

def to_mel(f: npt.ArrayLike):
    '''
    Convert linear frequency to mel scale.
    
    Source: https://en.wikipedia.org/wiki/Mel_scale
    '''
    
    return 2595*np.log10(1+f/700)