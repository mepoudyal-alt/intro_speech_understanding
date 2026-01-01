import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    num_frames = int((len(waveform) - frame_length + 1) // step)
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start_idx = i * step
        end_idx = min(start_idx + frame_length, len(waveform))
        frames[i, :end_idx-start_idx] = waveform[start_idx:end_idx]
    return frames

def VAD(waveform, Fs, threshold=0.1):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    # raise RuntimeError("You need to change this part")
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    frames = waveform_to_frames(waveform, frame_length, step)
    num_frames = frames.shape[0]

    energies = np.zeros(num_frames)
    for i in range(num_frames):
        energies[i] = np.sum(frames[i] ** 2)

    max_energy = np.max(energies)
    segments = []
    start = None

    for i in range(num_frames):
        if energies[i] > threshold * max_energy:
            if start is None:
                start = i
        else:
            if start is not None:
                segment = waveform[start * step:(i - 1) * step + frame_length]
                segments.append(segment)
                start = None

    if start is not None:
        segment = waveform[start * step:]
        segments.append(segment)

    return segments

def segments_to_models(segments, Fs, frame_length=None, step=None):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    # raise RuntimeError("You need to change this part")
    if frame_length is None:
        frame_length = int(0.004 * Fs)
    if step is None:
        step = int(0.002 * Fs)
    models = []

    for seg in segments:
        # Pre-emphasis
        preemph = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        # Frame
        frames = waveform_to_frames(preemph, frame_length, step)

        # Magnitude STFT
        mstft = np.abs(np.fft.fft(frames, axis=1))

        # Keep low-frequency half
        half = mstft[:, :frame_length // 2]

        # Log spectrum
        log_spec = 20 * np.log10(np.maximum(1e-12, half))

        # Average spectrum
        model = np.mean(log_spec, axis=0)
        models.append(model)

    return models

def recognize_speech(testspeech, Fs, models, labels, threshold=0.5):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    # raise RuntimeError("You need to change this part")
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs, frame_length=int(0.004*Fs), step=int(0.002*Fs))

    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    test_outputs = []

    for k in range(K):
        for y in range(Y):
            num = np.dot(models[y], test_models[k])
            den = np.linalg.norm(models[y]) * np.linalg.norm(test_models[k])
            sims[y, k] = num / den if den > 0 else 0

        best = np.argmax(sims[:, k])
        if sims[best, k] > threshold:
            test_outputs.append(labels[best])

    return sims, test_outputs

