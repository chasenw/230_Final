import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers

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

# Load the metadata
df = pd.read_csv('/Users/chasenwamu/Classes/CS230/kaggleSet/geotagged_images.csv')

# Split the DataFrame into training and testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(train_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((300, 300, 3), (2,))
).batch(32).prefetch(1)

test_dataset = tf.data.Dataset.from_generator(
    lambda: load_data(test_df),
    output_types=(tf.float32, tf.float32),
    output_shapes=((300, 300, 3), (2,))
).batch(32).prefetch(1)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(2)
])

# Compile the model
model.compile(optimizer='adam',
              loss=custom_mse_loss,
              metrics=['mean_squared_error'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
model.evaluate(test_dataset)
