"""
Segmentation using silence detection with spectral flatness of chroma features.
WiMIR workshop topic: Verse and chorus detection in vocal cover versions.
Author - Shreyan Chowdhury
"""

import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt


y, sr = librosa.core.load('/home/shreyan/PROJECTS/_data/structure_workshop/hero_vocals.wav')

chroma = librosa.feature.chroma_stft(y)

chroma_flatness = librosa.feature.spectral_flatness(S=chroma)

smoothed_chroma_flatness = np.convolve(chroma_flatness.squeeze(), np.ones(100))

bounds = librosa.segment.agglomerative(smoothed_chroma_flatness, 20)
xtimes = librosa.frames_to_time(range(len(smoothed_chroma_flatness)), sr=sr)

fig, (ax1, ax2) = plt.subplots(2, 1)
librosa.display.specshow(chroma, ax=ax1)

ax2.plot(xtimes, smoothed_chroma_flatness)
ax2.vlines(librosa.frames_to_time(bounds, sr=sr), 0, max(smoothed_chroma_flatness), color='black', linestyle='--',linewidth=2, alpha=0.9, label='Segment boundaries')
plt.show()
