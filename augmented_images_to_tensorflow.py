import tensorflow as tf

# Shuffle false because labels will be loaded in same format so have to be in same structure
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120))) # Make image smaller, neural network more efficient
train_images = train_images.map(lambda x: x / 255.0) # Lets us apply sigmoid activation to final layer of neural network
train_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255.0)
train_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255.0)

train_images.as_numpy_iterator().next()