import collections
import pathlib

import tensorflow as tf

from model import load_inception_v3

# Constants
IMAGE_SHAPE = (224, 224, 3)
MAX_CAPTION_LENGTH = 50
VOCABULARY_SIZE = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_SIZE = 128
UNITS = 128


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)  # load image from disk
    image = tf.io.decode_jpeg(image, channels=3)  # load as tensor
    image = tf.keras.layers.Resizing(299, 299)(image)  # resize
    image = tf.cast(image, tf.float32) / 255.0  # normalize
    return image


def extract_features(image_path, model):
    image = preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0)  # batch axis
    features = model(image)
    return image, features


def preprocess_caption(caption):
    pass


def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def generate_caption(image_path, model, word_to_index_map):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    image_feature = tf.expand_dims(preprocessed_image, axis=0)

    # Generate caption using the LSTM decoder
    caption = model.predict(image_feature)[0]
    predicted_caption = ''
    for word_index in caption:
        if word_index != 0:  # Skip padding (0)
            predicted_caption += word_to_index_map[word_index] + ' '
    predicted_caption = predicted_caption.strip()

    return predicted_caption
