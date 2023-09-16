# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:42:42 2023

@author: rayne
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import os

# 16 (E1) to 97 (C#8), 82 classes
processed_path = '../processing/processed'
X_train = np.array([]).reshape(0, 324, 5)
C_train = np.array([]).reshape(0, 12)
y_train = np.array([]).reshape(0, 98)
z_train = np.array([]).reshape(0)

for root, dirs, files in os.walk(processed_path + 'spec', topdown = False):
    for direc in dirs:   
        if not("_" in direc):
            continue
        path = os.path.join(root, direc)
        with open(path + "/X_train.npy", 'rb') as file:
            X = np.load(file)
        with open(path + "/C_train.npy", 'rb') as file:     
            C = np.load(file)
        with open(path + "/y_train.npy", 'rb') as file:
            y = np.load(file)
        with open(path + "/z_train.npy", 'rb') as file:
            z = np.load(file)
        X_train = np.append(X_train, X, axis = 0)
        C_train = np.append(C_train, C, axis = 0)
        y_train = np.append(y_train, y, axis = 0)
        z_train = np.append(z_train, z)
print("Training data loaded with shapes ", X_train.shape, C_train.shape, y_train.shape, z_train.shape)

min_note_number = 16

print(X_train[0][:, 0])
print(y_train[0])


#X_train, X_val, C_train, C_val, y_train, y_val, z_train, z_val = train_test_split(X_train, C_train, y_train, z_train, stratify = y_train, test_size = 0.1, random_state = 0)



model_path = '../model'
#model = tf.keras.models.load_model(model_path + '/model.keras')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_path + '/model_checkpoint/model_checkpoint',
    save_weights_only = True,
    monitor = 'accuracy',
    mode= 'max',
    save_best_only=True)



spec_model = models.Sequential()


spec_model.add(layers.Conv2D(64, (2, 1), activation='relu', input_shape = (324, 5, 1)))

spec_model.add(layers.Conv2D(256, (2, 1), activation='relu'))


spec_model.add(layers.Conv2D(512, (2, 1), activation='relu'))

spec_model.add(layers.Flatten())


spec_model.add(layers.Dense(128, activation='relu'))

chroma_model = models.Sequential()
chroma_model.add(layers.Dense(64, activation = 'relu', input_shape = (12,)))



mixed_concat_layer = layers.concatenate([spec_model.output, chroma_model.output])
mixed_hidden_layer = layers.Dense(64, activation="relu")(mixed_concat_layer)

mixed_hidden_layer = layers.Dense(128, activation="relu")(mixed_hidden_layer)

mixed_output_layer = layers.Dense(82, activation="sigmoid")(mixed_hidden_layer)

model = models.Model(inputs = [spec_model.input, chroma_model.input], 
                            outputs = mixed_output_layer)
model.summary()


model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits = False, apply_class_balancing = True),
              metrics = 'accuracy')

history = model.fit([X_train, C_train], 
                    y_train[:, min_note_number:],
                   # validation_data = ([X_val, C_val], y_val[:, min_note_number:]),
                    epochs = 10,
                    callbacks = [model_checkpoint_callback])

model.load_weights(model_path + '/model_checkpoint/model_checkpoint')
model.save(model_path + '/model.keras')

tf.keras.utils.plot_model(
    model,
    to_file = model_path + '/model.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96,
    show_layer_activations=True,
    show_trainable=True
)



tf.keras.backend.clear_session()
del model