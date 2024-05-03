import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial
import numpy as np
import pysptk
import pyworld as pw
from fastdtw import fastdtw
from dtw import dtw
from scipy import spatial
from scipy.signal import medfilt
import matplotlib.pyplot as plt

import typing
import numpy.typing as npt

def pitch_contour_pruned_depreciated(filename: str, to_prune : bool = False) :
    '''
    Extract pitch contour from a wav file using WORLD Vocoder.
    filename: str, the name of the wav file.
    to_prune: bool, determine whether to remove the voiceless frames.
    
    It is depreciated because the function is not optimized! Please use world_extract instead.
    '''
    
    y, fs = sf.read(filename)
    
    assert fs == 16000, "Sampling rate must be 16000"
    assert len(y.shape) == 1, "Only mono audio files are supported"
    
    f0, _, _ = pw.wav2world(y, fs)
    
    f0[f0 > 500] = 0 # remove outliers
    
    f0 = medfilt(f0, 3) # median filtering
    
    if to_prune:
        f0 = f0[f0 != 0]
    
    return f0

def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    
    """
    Extract World-based acoustic features.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0

def _get_best_mcep_params(fs: int) -> typing.Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")

def to_mel(f: npt.ArrayLike):
    '''
    Convert linear frequency to mel scale.
    
    Source: https://en.wikipedia.org/wiki/Mel_scale
    '''
    
    return 2595*np.log10(1+f/700)

def pitch_diff_dtw(pitch_learner: npt.ArrayLike, pitch_instuctor: npt.ArrayLike, visualize : bool = False) :
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