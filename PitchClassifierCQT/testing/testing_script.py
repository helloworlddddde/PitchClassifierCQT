# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:37:10 2023

@author: rayne
"""

import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt

# 16 (E1) to 97 (C#8), 82 classes

processed_path = '../processing/processed'




X_train = np.array([]).reshape(0, 324, 5)
C_train = np.array([]).reshape(0, 12)
y_train = np.array([]).reshape(0, 98)
z_train = np.array([]).reshape(0)
X_test = np.array([]).reshape(0, 324, 5)
C_test = np.array([]).reshape(0, 12)
y_test= np.array([]).reshape(0, 98)
z_test = np.array([]).reshape(0)
for note_index in range(16, 98):
    with open(processed_path + "spec/" + str(note_index) + "/X_train.npy", 'rb') as f:
        X_train_curr = np.load(f)
        X_train = np.append(X_train, X_train_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/C_train.npy", 'rb') as f:
        C_train_curr = np.load(f)
        C_train = np.append(C_train, C_train_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/y_train.npy", 'rb') as f:
        y_train_curr = np.load(f)
        y_train = np.append(y_train, y_train_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/z_train.npy", 'rb') as f:
        z_train_curr = np.load(f)
        z_train = np.append(z_train, z_train_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/X_test.npy", 'rb') as f:
        X_test_curr = np.load(f)
        X_test = np.append(X_test, X_test_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/C_test.npy", 'rb') as f:
        C_test_curr = np.load(f)
        C_test = np.append(C_test, C_test_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/y_test.npy", 'rb') as f:
        y_test_curr = np.load(f)
        y_test = np.append(y_test, y_test_curr, axis = 0)
    with open(processed_path + "spec/" + str(note_index) + "/z_test.npy", 'rb') as f:
        z_test_curr = np.load(f)
        z_test = np.append(z_test, z_test_curr, axis = 0)
print("Training and test data loaded")


min_note_number = 16

model_path = '../model'
model = tf.keras.models.load_model(model_path + '/model.keras')

model.summary()
model.evaluate([X_train, C_train], y_train[:, min_note_number:])
model.evaluate([X_test, C_test], y_test[:, min_note_number:])


tf.keras.backend.clear_session()
del model