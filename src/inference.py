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
inputs = pd.read_hdf(dataset_dir+'test.h5', 'images').values.reshape(-1,img_size,img_size,channels)
labels = pd.read_hdf(dataset_dir+'test.h5', 'labels').values.reshape(-1,1)
print('*************** Data read')

ground_truths = K.utils.to_categorical(labels, num_classes=classes)
inputs = inputs.astype('float32')
inputs /= 65535

# Load model 
model = K.models.load_model(model_dir+model_name+'.h5')
model.summary()
print("Model loaded!")

# Evaluate loss function and accuracy
results = model.evaluate(inputs, ground_truths)
print('[Loss, Accuracy] =', results)

print('Evaluation completed!')