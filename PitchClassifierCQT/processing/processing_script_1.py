# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 08:09:42 2023

@author: rayne
"""
import os
import utils.featureextractor as fe
import soundfile
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split


raw_path = 'raw'
processed_path = 'processed'
sr = 44.1E3
X_train = np.array([])
C_train = np.array([])
y_train = np.array([])
z_train = np.array([])
X_test = np.array([])
C_test = np.array([])
y_test = np.array([])
z_test = np.array([])


for note_index in range(16, 98):
    curr_X = []
    curr_C = []
    curr_y = []
    curr_z = []
    idx = 0
    

        
    for root, dirs, files in os.walk(raw_path, topdown = False):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            note_letter = file_name.split("-")[2]
            note_number = fe.get_note_number(note_letter)
            
            if note_number == note_index:
                with soundfile.SoundFile(file_path) as sound_file:
                    sound_data = sound_file.read(dtype="float32")
                    sound_data = sound_data / np.max(np.abs(sound_data))
                    for i in range(0, sound_data.shape[0] // int(sr / 5) - 14):
                        
                        if i >= 75:
                            break
                        
                        segment = sound_data[i * int(sr / 5) : (i + 1) * int(sr / 5)]
                        
                        if not os.path.exists(processed_path + "/" + str(note_index)):
                            os.makedirs(processed_path + "/" + str(note_index))
                        
                        soundfile.write(processed_path + "/" + str(note_index) + "/" + str(idx) + '-' + file_name,
                                        segment,
                                        int(sr))
                        
                        
                        idx += 1
                        
                        
                        
                        
            
    if idx == 0:
        break
    

    
    

 
        