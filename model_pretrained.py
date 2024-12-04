import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(path):
    """Load an image from a file path and preprocess it."""
    image = Image.open(path).resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    return image_array

def load_data(df):
    """Generator function to load and preprocess images and labels."""
    for _, row in df.iterrows():
        image_path = row['file_path']
        image = load_and_preprocess_image(image_path)
        label = [row['latitude'], row['longitude']]
        yield image, label
def degrees_to_radians(degrees):
    """Convert degrees to radians using TensorFlow operations."""
    pi = tf.constant(np.pi, dtype=tf.float32)  # Directly define pi with the correct dtype
    return degrees * (pi / 180.0)


def haversine_loss(y_true, y_pred):
    """Calculate the Haversine loss between true and predicted geolocations."""
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    lat1, lon1 = degrees_to_radians(lat_true), degrees_to_radians(lon_true)
    lat2, lon2 = degrees_to_radians(lat_pred), degrees_to_radians(lon_pred)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = tf.sin(dlat / 2.0)**2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0)**2
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    return tf.reduce_mean(6371 * c)  # Earth radius in kilometers

# Load the metadata
df = pd.read_csv('/content/geotagged_images.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Use MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Create new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(2)(x)  # Output layer for latitude and longitude
model = models.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=haversine_loss,
              metrics=['mean_squared_error'])

# Prepare the data
train_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(train_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((224, 224, 3), (2,))
).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(test_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((224, 224, 3), (2,))
).batch(32).prefetch(tf.data.AUTOTUNE)

# Determine the number of steps per epoch
num_train_samples = len(train_df)
num_test_samples = len(test_df)
batch_size = 32
steps_per_epoch = np.ceil(num_train_samples / batch_size).astype(int)
validation_steps = np.ceil(num_test_samples / batch_size).astype(int)

log_dir = "logs/training_run"
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

#tensorboard stuff
%load_ext tensorboard
%tensorboard --logdir logs/training_run

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[tensorboard_callback])

# Evaluate the model
model.evaluate(test_dataset)
