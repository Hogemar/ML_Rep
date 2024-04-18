import os
from os.path import isfile, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged, 3 - INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

import numpy as np
import random

import cv2

from keras.regularizers import l2

folder_train = "Faces"
folder_src = "Input"
folder_dst = "Output"
model_file = "ss.weights.h5"
brightness_corr = 250
do_train = True

# Get train images
X = []
for filename in os.listdir(folder_train):
    X.append(img_to_array(load_img(folder_train + os.sep + filename)))

X = np.array([cv2.resize(i, (256, 256)) for i in X], dtype=float)/255.0
# X = np.array(X, dtype=float)
Xtrain = X

# Model
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

if do_train:
    # Train model
    model.fit(image_a_b_gen(batch_size), epochs=2, steps_per_epoch=50)

    # Save model
    model.save_weights(model_file)

# Load model
model.load_weights(model_file)

# Process images
color_me = []
onlyfiles = [f for f in os.listdir(folder_src) if isfile(join(folder_src, f))]
for filename in onlyfiles:
    color_me.append(img_to_array(load_img(folder_src + os.sep + filename)))
color_me = np.array([cv2.resize(i, (256, 256)) for i in color_me], dtype=float)
# color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i] * 128
    img_rgb = lab2rgb(cur)*brightness_corr
    imsave(folder_dst + os.sep + "img_%d.png" % i, img_rgb.astype(np.uint8))
    print("img_%d.png saved" % i)