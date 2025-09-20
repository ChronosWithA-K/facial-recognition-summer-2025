# Code is broken (this is intentional, contents were moved to data/train/images and data/train/labels)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stops floating-point round-off errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress '[...] rebuild TensorFlow with the appropriate compiler flags'

import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
import keras

images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False)
# print(images.as_numpy_iterator().next()) # Test images appear - they do

def load_image(image_path):
    byte_img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)

# Print RGB channels
for img in images:
    print(img.numpy())

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

# Show batches of 4 random images
fix, ax = plt.subplots(1, 4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
