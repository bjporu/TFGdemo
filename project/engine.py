import torch
import numpy as np
import soundfile as sf
import librosa
import ruptures as rpt
from IPython.display import clear_output

from chord_prediction import CNN, predict
from segmentation import segmentate

# Global
SAMPLERATE = 44100  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_mapping =  { "NC": 0, "A": 1, "A#": 2, "B": 3, "C": 4, "C#": 5, "D": 6, "D#": 7, "E": 8, "F": 9, "F#": 10, "G": 11, "G#": 12 }
chord_mapping = { "NC": 0, "": 1, "m": 2, "5": 3, "7": 4, "maj7": 5, "m7": 6, "6": 7, "m6": 8, "9": 9, "m9": 10, "dim": 11, "aug": 12, "sus2": 13, "sus4": 14, "m7b5": 15 }

def load_models():

    root_model = CNN(num_classes=len(root_mapping))
    root_model.load_state_dict(torch.load('rootcqtcnn_asim1_1005.pth', map_location=device))
    root_model.eval()

    chord_model = CNN(num_classes=len(chord_mapping))
    chord_model.load_state_dict(torch.load('chordcqtcnn_asim1_0905.pth', map_location=device))
    chord_model.eval()

    return root_model, chord_model

def main(audio_object, sr, criteria):
    
    print("Detecting Onsets")

    audio_data, sr = librosa.load(audio_object, sr=SAMPLERATE)
    audio_data = audio_data.flatten()
    final_onsets = segmentate(audio_data, sr, criteria)

    print("Estimating Chords")

    root_model, chord_model = load_models()

    chord_predictions = []

    for i in range(len(final_onsets) - 1):

        start, end = int(final_onsets[i] * sr), int(final_onsets[i+1] * sr)
        window = audio_data[start:end]

        prediction, root, chord = predict(window, root_model, chord_model, root_mapping, chord_mapping, device)
        chord_predictions.append(prediction)

    # Mask onsets that fall too close given a threshold
    min_duration = 0.5
    durations = np.diff(final_onsets)
    mask = durations > min_duration

    # Apply mask to both onset intervals and predictions
    filtered_onsets = final_onsets[:-1][mask]
    filtered_preds = [chord_predictions[i] for i, keep in enumerate(mask) if keep]

    # Add last onset for display alignment
    filtered_onsets = np.append(filtered_onsets, final_onsets[-1])

    # Ordenarlo todo en una lista para el JSON 
    annotations = [
        {
            "start": float(filtered_onsets[i]),
            "end": float(filtered_onsets[i + 1]),
            "chord": str(filtered_preds[i])
        }
        for i in range(len(filtered_preds))
    ]

    return annotations