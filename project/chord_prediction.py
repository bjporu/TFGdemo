import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa 
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1 - asymmetric kernel to capture harmonic structure
            nn.Conv2d(1, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [batch, 32, 42, 215]

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [batch, 64, 21, 107]

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # → [batch, 128, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def predict(window, root_model, chord_model, root_mapping, chord_mapping, device, sr=44100, max_length=5.0):
    # audio_data = audio_data.flatten()

    if len(window) > max_length * sr:
        window = window[:int(max_length*sr)]

    elif len(window) < max_length * sr:
        window = np.pad(window, (0, int(max_length * sr) - len(window)))

    C = librosa.cqt(window, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
    C = librosa.amplitude_to_db(abs(C))
    C = (C - np.mean(C)) / (np.std(C) + 1e-8)

    torch_cqt = torch.tensor(np.array(C), dtype=torch.float32)
    torch_cqt = torch_cqt.unsqueeze(0).unsqueeze(0)
    torch_cqt = torch_cqt.to(device)

    with torch.no_grad():
        root_output = root_model(torch_cqt)
        predicted_root_label = torch.argmax(root_output, dim=1).item()

        chord_output = chord_model(torch_cqt)
        predicted_chord_label = torch.argmax(chord_output, dim=1).item()

    root_label_mapping = {v: k for k, v in root_mapping.items()}
    predicted_root = root_label_mapping[predicted_root_label]

    chord_label_mapping = {v: k for k, v in chord_mapping.items()}
    predicted_chord = chord_label_mapping[predicted_chord_label]

    if predicted_root == "NC" or predicted_chord == "NC": result = "NC"
    else : result = predicted_root + predicted_chord

    return result, predicted_root, predicted_chord