import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    if f <= 0 or Fs <= 0:
        raise ValueError("f and Fs must be positive")
    n = int(Fs/2)
    t = np.arange(n)/Fs
    f_third = f * (2**(4/12))
    f_fifth = f * (2**(7/12))
    x = np.cos(2*np.pi*f*t) + np.cos(2*np.pi*f_third*t) + np.cos(2*np.pi*f_fifth*t)
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    if N <= 0:
        raise ValueError("N must be positive")
    j = 1j
    W = np.zeros((N,N), dtype='complex')
    for k in range(N):
        for n in range(N):
            W[k,n] = np.cos(2*np.pi*k*n/N) - j*np.sin(2*np.pi*k*n/N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    if len(x) == 0 or Fs <= 0:
        raise ValueError("x must be non-empty and Fs must be positive")
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1/Fs)

    # Keep only non-negative frequencies
    pos_mask = f >= 0
    Xp = np.abs(X[pos_mask])
    fp = f[pos_mask]

    # Find indices of three largest peaks
    idx = np.argsort(Xp)[-3:]
    freqs = np.sort(fp[idx])

    return freqs[0], freqs[1], freqs[2]

