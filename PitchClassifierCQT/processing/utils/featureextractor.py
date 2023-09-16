# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:18:32 2023

@author: rayne
"""

import librosa
import numpy as np




def extract_chroma(sound_data, sr):
    S = np.abs(librosa.cqt(sound_data, sr = sr, n_bins = 108 * 3, hop_length = 8192 // 4, bins_per_octave = 12 * 3))
    

    S = S / np.max(S)
    chroma = librosa.feature.chroma_cqt(C = S, sr = sr, bins_per_octave = 36, n_octaves = 8)
    chroma = chroma
    f_class = np.linspace(1, 12, 12)
    duration = sound_data.size / sr
    t_scale = np.linspace(0, duration, chroma[0, :].size)
    
    return chroma, f_class, t_scale

def extract_spec(sound_data, sr):
    S = np.abs(librosa.cqt(sound_data, sr = sr, n_bins = 108 * 3, hop_length = 8192 // 4, bins_per_octave = 12 * 3))
    spec = S / np.max(S)
    f_scale= np.linspace(0, sr/2, spec[:, 0].size)
    duration = sound_data.size / sr
    t_scale = np.linspace(0, duration, spec[0, :].size)
    return spec, f_scale, t_scale
 
def get_mean_chroma(chroma):
    chroma = chroma / np.max(chroma)
    mean_chroma = np.mean(chroma, axis = 1)
    return mean_chroma

def get_note_letter(note_number):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = note_number
    octave = 0
    while n >= 12:
        octave += 1
        n -= 12
    return notes[n] + str(octave)

def get_note_number(note_letter):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = notes.index(note_letter[0:-1])
    return n + (int(note_letter[-1]) * 12)



