import unittest
import numpy as np
import librosa

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    if not isinstance(frame_length, int) or frame_length <= 0:
        raise ValueError('frame_length must be a positive integer')
    if not isinstance(step, int) or step <= 0:
        raise ValueError('step must be a positive integer')
    num_frames = int(np.ceil((len(waveform)-frame_length+1)/float(step)))
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start_idx = i*step
        end_idx = min(start_idx + frame_length, len(waveform))
        frames[i,:end_idx-start_idx] = waveform[start_idx:end_idx]
    return frames
speech, Fs = librosa.load('train.m4a', sr=None)

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    if not isinstance(frames, np.ndarray) or frames.size == 0:
        raise ValueError('frames must be a non-empty matrix')
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
    if not isinstance(mstft, np.ndarray) or mstft.size == 0:
        raise ValueError('mstft must be a non-empty matrix')
    spectrogram = 20*np.log10(np.maximum(0.001*np.amax(mstft),mstft))
    return spectrogram

