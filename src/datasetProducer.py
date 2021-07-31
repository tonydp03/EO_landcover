'''
Basic code to read the EuroSAT images and create a dataframe with images and labels, for fast access during training and inference.
Each row of the dataframe corresponds to an image (if opening "images") or the label - class (if opening "labels")
'''

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle 
import tifffile
from sklearn.utils import shuffle 


dataset_dir = '../EuroSAT_AllBands/'
save_dir = '../datasets/'
dfdata_dir = '../df_data/'

os.makedirs(dfdata_dir, exist_ok=True)

list_dir = os.listdir(dataset_dir)

train_images = []
train_labels = []
test_images = []
test_labels = []

label_legend = []

for f in enumerate(list_dir):
    imgs =[]
    lbls = []
    folder_dir = dataset_dir + f[1]+ '/'
    images_in_folder = os.listdir(folder_dir)
    for im in images_in_folder:
        imarray = np.asarray(tifffile.imread(folder_dir+im))
        imgs.append(imarray)
    imgs = shuffle(imgs)
    lbls = len(imgs)*[f[0]]
    train_images = train_images + imgs[250:]
    test_images = test_images + imgs[:250]
    train_labels = train_labels + lbls[250:]
    test_labels = test_labels + lbls[:250]

    label_legend.append(f)


# Save legend of numbers corresponding to classes
with open(r'../label_legend.txt', 'w') as ftxt:
    ftxt.writelines(str(label_legend))

# Create inputs and labels as arrays
train_inputs = np.asarray(train_images, dtype='float32')
train_labels = np.asarray(train_labels, dtype='float32')
test_inputs = np.asarray(test_images, dtype='float32')
test_labels = np.asarray(test_labels, dtype='float32')
   

# Now shuffle data
train_inputs, train_labels = shuffle(train_inputs, train_labels)
test_inputs, test_labels = shuffle(test_inputs, test_labels)

# Flatten arrays and save them as dataframes 
flat_train_inputs = train_inputs.reshape(train_inputs.shape[0], -1)
flat_test_inputs = test_inputs.reshape(test_inputs.shape[0], -1)


df_train_images = pd.DataFrame(data=flat_train_inputs)
df_train_labels = pd.DataFrame(data=train_labels)
df_test_images = pd.DataFrame(data=flat_test_inputs)
df_test_labels = pd.DataFrame(data=test_labels)

# Convert to h5 and save
df_train_images.to_hdf(dfdata_dir+"train.h5","images",append=False)
df_train_labels.to_hdf(dfdata_dir+"train.h5","labels",append=False)
df_test_images.to_hdf(dfdata_dir+"test.h5","images",append=False)
df_test_labels.to_hdf(dfdata_dir+"test.h5","labels",append=False)

print("Dataset saved!")

