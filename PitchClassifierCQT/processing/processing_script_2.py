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
import librosa


processed_path = 'processed'
sr = 44.1E3
ivs = []
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
for note_index in range(16, 98):
    curr_X = []
    curr_C = []
    curr_y = []
    curr_z = []

    for root, dirs, files in os.walk(processed_path + '/' + str(note_index), topdown = False):
        
        sounds1 = []
        names1 = []
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            with soundfile.SoundFile(file_path) as sound_file:
                sound_data = sound_file.read(dtype="float32")
                sound_data = sound_data / np.max(np.abs(sound_data))
                sounds1.append(sound_data)
                names1.append(file_name[:-4])
                

    
    
    sounds1 = np.array(sounds1)
    names1 = np.array(names1)
    
    
    N1 = sample_without_replacement(sounds1.shape[0], min(sounds1.shape[0], 150), random_state = 0)
    
    segments1 = sounds1[N1]
    segmentnames1 = names1[N1]
    
    
    for iv in ivs:
        
        for root, dirs, files in os.walk(processed_path + '/' + str(note_index + iv), topdown = False):
            
            sounds2 = []
            names2 = []
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                with soundfile.SoundFile(file_path) as sound_file:
                    sound_data = sound_file.read(dtype="float32")
                    sound_data = sound_data / np.max(np.abs(sound_data))
                    sounds2.append(sound_data)
                    names2.append(file_name[:-4])
            
        
        
        sounds2 = np.array(sounds2)
        names2 = np.array(names2)
        
        
        
        N2 = sample_without_replacement(sounds2.shape[0], min(sounds2.shape[0], 150), random_state = 0)
        
        segments2 = sounds2[N2]
        segmentnames2 = names2[N2]
    
    
        segments_1 = segments1[:min(segments1.shape[0], segments2.shape[0])]
        
        segments_2 = segments2[:min(segments1.shape[0], segments2.shape[0])]
        
        segments1 = segments_1 + segments_2
        
        segments1 = segments1 / np.max(segments1, axis = 0)
    
    
        segmentnames_1 = segmentnames1[:min(segments1.shape[0], segments2.shape[0])]
        segmentnames_2 = segmentnames2[:min(segments1.shape[0], segments2.shape[0])]
        
        segmentnames1 = np.char.add(np.char.add(segmentnames_1, '_'), segmentnames_2)
    
        
    
    sound_path = processed_path + "/" + str(note_index)
    
    for iv in ivs:
        sound_path = sound_path + "_" + str(note_index + iv)
        
    for k in range(segments1.shape[0]):
        segment = segments1[k]
        segment_name = segmentnames1[k]
        spec, f, t = fe.extract_spec(segment, sr)
        chroma, f_class, t = fe.extract_chroma(segment, sr)
        mean_chroma = fe.get_mean_chroma(chroma)
        curr_X.append(spec)
        curr_C.append(mean_chroma)
        y_temp = np.zeros((98,))
        y_temp[note_index] = 1
        
        for iv in ivs:   
            y_temp[note_index + iv] = 1
        
        curr_y.append(y_temp)
        curr_z.append(segment_name)
        
      
        if not os.path.exists(sound_path):
            os.makedirs(sound_path)
            
        
        #soundfile.write(sound_path + "/" + segment_name + '.wav', segment, int(sr))
        

        
        
    # librosa.display.specshow(spec, bins_per_octave = 36, fmax = librosa.note_to_hz('C10'), x_axis = 'time', y_axis = 'cqt_hz',
    #                           sr = sr,
    #                           hop_length = 8192 // 4)
   
    
    curr_X = np.array(curr_X)
    curr_C = np.array(curr_C)
    curr_y = np.array(curr_y)
    curr_z = np.array(curr_z)
    
    curr_X_train, curr_X_test, curr_C_train, curr_C_test, curr_y_train, curr_y_test, curr_z_train, curr_z_test = train_test_split(curr_X, curr_C, curr_y, curr_z, test_size = 0.2, random_state = 0)
    
    print(note_index)
    print(curr_X_train.shape, curr_X_test.shape)
    print(curr_C_train.shape, curr_C_test.shape)
    print(curr_y_train.shape, curr_y_test.shape)
    print(curr_z_train.shape, curr_z_test.shape)
    
    save_path = processed_path + "spec/" + str(note_index)
    
    for iv in ivs:
        save_path = save_path + "_" + str(note_index + iv) 
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(save_path + "/X_train.npy", 'wb') as file:
        np.save(file, curr_X_train)
    with open(save_path + "/C_train.npy", 'wb') as file:     
        np.save(file, curr_C_train)
    with open(save_path + "/y_train.npy", 'wb') as file:
        np.save(file, curr_y_train)
    with open(save_path + "/z_train.npy", 'wb') as file:
        np.save(file, curr_z_train)
    with open(save_path + "/X_test.npy", 'wb') as file:
        np.save(file, curr_X_test)
    with open(save_path + "/C_test.npy", 'wb') as file:     
        np.save(file, curr_C_test)
    with open(save_path +  "/y_test.npy", 'wb') as file:
        np.save(file, curr_y_test)
    with open(save_path +  "/z_test.npy", 'wb') as file:
        np.save(file, curr_z_test)
    



    
    

 
        