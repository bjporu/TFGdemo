import librosa
import numpy as np
import ruptures as rpt
import torch

def segmentate(audio_data, sr, criteria):
 
    harm, perc = librosa.effects.hpss(audio_data)

    if criteria == "cosdif":
        cqt = librosa.cqt(y=harm, sr=sr, n_bins=84, bins_per_octave=12)
        cqt = librosa.amplitude_to_db(abs(cqt))
        cqt = (cqt - np.mean(cqt)) / (np.std(cqt) + 1e-8)

        cqt_smooth = librosa.decompose.nn_filter(cqt, aggregate=np.median, metric='cosine')

        cpd = rpt.Pelt(model="rbf").fit(cqt_smooth.T) # Change-Point Detection
        breakpoints = cpd.predict(pen=10)  # Ajustar 

        frame_times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr, hop_length=512)
        output_onsets = librosa.frames_to_time(breakpoints[:-1], sr=sr, hop_length=512)

    elif criteria == "tempo":
        tempo, beat_frames = librosa.beat.beat_track(y=perc, sr=sr, tightness=150) #, trim=False, tightness=100, units='frames')
        print(f"Tempo: {(tempo[0]):.1f}bpm")

        if tempo > 100: output_onsets = librosa.frames_to_time(beat_frames[::2], sr=sr)
        else: output_onsets = librosa.frames_to_time(beat_frames, sr=sr)

    final_onsets = np.unique(np.concatenate(([0.0], np.array(output_onsets), [np.floor(len(audio_data)/sr)])))

    return final_onsets

