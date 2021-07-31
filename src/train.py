'''
Code to perform training of a CNN with the EuroSAT dataset.
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import modelsCNN
import pandas as pd

batch_size = 64
img_size = 64
channels = 13
classes = 10
epochs = 25
dataset_dir = '../df_data/'
model_dir = '../models/'
hist_dir = model_dir + 'histories/'

model_name = 'basic_CNN'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)

print('*************** Start to read data')
# Read data
inputs = pd.read_hdf(dataset_dir+'train.h5', 'images').values.reshape(-1,img_size,img_size,channels)
labels = pd.read_hdf(dataset_dir+'train.h5', 'labels').values.reshape(-1,1)
print('*************** Data read')

ground_truths = K.utils.to_categorical(labels, num_classes=classes)
inputs = inputs.astype('float32')
inputs /= 65535

# Create model 
model = getattr(modelsCNN, model_name)([img_size,img_size,channels], classes)
model.summary()

# Train the model
history = model.fit(inputs, ground_truths, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save training information on accuracy and loss function
history_save = pd.DataFrame(history.history).to_hdf(hist_dir + model_name + '_history.h5', "history", append=False)

# Save model and weights
model.save(model_dir + model_name)
print('Trained model saved @ %s ' % model_dir)
