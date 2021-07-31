'''
Convolutional neural network for land cover/classification with the EuroSAT dataset.
Additional models can be put here.
'''


import tensorflow as tf
from tensorflow import keras as K


def basic_CNN(input_shape, classes):
    nb_filter = [32, 64, 128]
    kernel_size = (3,3)
    L2_reg = K.regularizers.l2(1e-4)
    lossfunc='categorical_crossentropy'

    input_img = K.Input(shape=input_shape, name = 'input_layer')

    conv1 = K.layers.Conv2D(nb_filter[0], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv1_1')(input_img)
    conv1 = K.layers.Conv2D(nb_filter[0], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv1_2')(conv1)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
    drop1 = K.layers.Dropout(0.25, name='drop1')(pool1)

    conv2 = K.layers.Conv2D(nb_filter[1], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv2_1')(drop1)
    conv2 = K.layers.Conv2D(nb_filter[1], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv2_2')(conv2)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
    drop2 = K.layers.Dropout(0.25, name='drop2')(pool2)

    conv3 = K.layers.Conv2D(nb_filter[2], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv3_1')(drop2)
    conv3 = K.layers.Conv2D(nb_filter[2], kernel_size, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='conv3_2')(conv3)
    pool3 = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(conv2)
    drop3 = K.layers.Dropout(0.25, name='drop3')(pool3)

    flat = K.layers.Flatten()(drop3)
    dense1 = K.layers.Dense(512, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='dense1')(flat)
    dense2 = K.layers.Dense(128, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='dense2')(dense1)
    output = K.layers.Dense(classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=L2_reg, name='output_layer')(dense2)

    model = K.Model(inputs=input_img, outputs=output, name='Basic_CNN')
    model.compile(loss=lossfunc, optimizer=K.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model