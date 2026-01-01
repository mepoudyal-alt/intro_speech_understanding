import numpy as np
import torch

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    emphasized = waveform - 0.97 * np.roll(waveform, 1)
    emphasized[0] = waveform[0]
    
    frame_length = int(0.004 * Fs)
    frame_step = int(0.002 * Fs)
    
    nframes = (len(emphasized) - frame_length) // frame_step + 1
    spectrogram = np.zeros((nframes, frame_length // 2 + 1))
    
    for i in range(nframes):
        frame = emphasized[i * frame_step : i * frame_step + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        windowed = frame * np.hamming(frame_length)
        spectrogram[i, :] = np.abs(np.fft.rfft(windowed))
    
    features = spectrogram[:, :spectrogram.shape[1] // 2]
    
    vad_frame_length = int(0.025 * Fs)
    vad_frame_step = int(0.010 * Fs)
    
    vad_frames = (len(emphasized) - vad_frame_length) // vad_frame_step + 1
    vad_energy = np.zeros(vad_frames)
    
    for i in range(vad_frames):
        frame = emphasized[i * vad_frame_step : i * vad_frame_step + vad_frame_length]
        if len(frame) < vad_frame_length:
            frame = np.pad(frame, (0, vad_frame_length - len(frame)))
        vad_energy[i] = np.sum(frame ** 2)
    
    threshold = np.percentile(vad_energy, 10)
    voice_activity = vad_energy > threshold
    
    segments = []
    in_segment = False
    segment_start = 0
    
    for i in range(len(voice_activity)):
        if voice_activity[i] and not in_segment:
            segment_start = i
            in_segment = True
        elif not voice_activity[i] and in_segment:
            segments.append((segment_start, i - 1))
            in_segment = False
    
    if in_segment:
        segments.append((segment_start, len(voice_activity) - 1))
    
    labels = np.zeros(len(features), dtype=int)
    
    for segment_num, (seg_start, seg_end) in enumerate(segments):
        start_frame = int(seg_start * vad_frame_step / frame_step)
        end_frame = int(seg_end * vad_frame_step / frame_step)
        
        end_frame = min(end_frame, len(features) - 1)
        
        if start_frame < len(features):
            label_idx = segment_num % 6
            labels[start_frame:end_frame+1] = label_idx
    
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    nfeats = features.shape[1]
    nlabels = 1 + int(np.max(labels))
    
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(nfeats),
        torch.nn.Linear(nfeats, nlabels)
    )
    
    features_tensor = torch.from_numpy(features).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    lossvalues = np.zeros(iterations)
    
    for i in range(iterations):
        outputs = model(features_tensor)
        loss = loss_function(outputs, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()
    
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    features_tensor = torch.from_numpy(features).float()
    
    with torch.no_grad():
        outputs = model(features_tensor)
    
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    
    return probabilities.detach().numpy()

