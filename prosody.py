import numpy as np
import wave
import matplotlib.pyplot as plt
import soundfile as sf

import pyworld as pw
from dtw import dtw

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

def prosodic_metric(pitch_learner: npt.ArrayLike, pitch_instuctor: npt.ArrayLike, visualize : bool = False) :
    '''
    Calculate the prosodic metric between the pitch contour of the learner and the instructor based on 
    the dynamic time warping algorithm.
    
    pitch_learner: npt.ArrayLike, the pitch contour of the learner.
    pitch_instuctor: npt.ArrayLike, the pitch contour of the instructor.
    visualize: bool, determine whether to visualize the alignment.
    '''
    
    f0_mel_learner = to_mel(pitch_learner)
    f0_mel_instuctor = to_mel(pitch_instuctor)
    
    f0_diff_learner = np.diff(f0_mel_learner)
    f0_diff_instuctor = np.diff(f0_mel_instuctor)
    
    alignment = dtw(f0_diff_learner, f0_diff_instuctor, keep_internals=False)
    
    if visualize:
        alignment.plot(type="twoway")
        plt.legend([f'dist: {alignment.distance:.2f}'])
        plt.xlabel('Instructor (frame index)')
        plt.ylabel('Student (frame index)')  

    return alignment.distance