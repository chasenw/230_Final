import tensorflow as tf
from tensorflow import math as tfm

import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks

def load_and_preprocess_image(path):
    """Load an image from a file path and preprocess it."""
    image = Image.open(path)
    image = image.resize((300, 300))  # Resize image
    image_array = np.array(image) / 255.0  # Convert image to array and normalize
    return image_array

def load_data(df):
    """Generator function to load and preprocess images and labels."""
    for _, row in df.iterrows():
        image_path = row['file_path']
        image = load_and_preprocess_image(image_path)
        label = [row['latitude'], row['longitude']]
        yield image, label

def custom_mse_loss(y_true, y_pred):
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    lat_loss = tf.reduce_mean(tf.square(lat_true - lat_pred))
    lon_loss = tf.reduce_mean(tf.square(lon_true - lon_pred))
    return lat_loss + lon_loss

def degrees_to_radians(degrees):
    """Convert degrees to radians using TensorFlow operations."""
    pi = tf.constant(np.pi, dtype=tf.float32)  # Directly define pi with the correct dtype
    return degrees * (pi / 180.0)

def haversine_loss(y_true, y_pred):
    """Calculate the Haversine loss between true and predicted values."""
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]

    # Convert degrees to radians
    lat1, lon1 = degrees_to_radians(lat_true), degrees_to_radians(lon_true)
    lat2, lon2 = degrees_to_radians(lat_pred), degrees_to_radians(lon_pred)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = tf.sin(dlat / 2.0)**2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0)**2
    c = 2 * tf.asin(tf.sqrt(a))
    km = 6371 * c  # Radius of the Earth in kilometers. Adjust the radius as per your need
    return tf.reduce_mean(km)




def haversine_metric(y_true, y_pred):
    # Reuse the haversine loss calculation to report distance errors
    return haversine_loss(y_true, y_pred)


import tensorflow as tf
print(tf.__version__)

#Confirm GPU access
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the metadata
df = pd.read_csv('/content/geotagged_images.csv')

log_dir = "logs/training_run"
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

# Split the DataFrame into training and testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(train_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((300, 300, 3), (2,))
).batch(64).repeat().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(test_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((300, 300, 3), (2,))
).batch(64).prefetch(tf.data.AUTOTUNE)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)  # Output layer for latitude and longitude
    #Potentially add additional laters to take out bias, but
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),  # 0.001 default, we can increase
              loss=haversine_loss,
              metrics=['mean_squared_error'])

#tensorboard stuff
%load_ext tensorboard
%tensorboard --logdir logs/training_run

# Train the model
num_train_samples = len(train_df)
num_test_samples = len(test_df)
batch_size = 64  # you've already set this
steps_per_epoch = np.ceil(num_train_samples / batch_size).astype(int)
steps_validation=np.ceil(num_test_samples/batch_size).astype(int)
model.fit(train_dataset, epochs=6, validation_data=test_dataset, steps_per_epoch=steps_per_epoch, validation_steps=steps_validation, callbacks=[tensorboard_callback])

# Evaluate the model
model.evaluate(test_dataset)
