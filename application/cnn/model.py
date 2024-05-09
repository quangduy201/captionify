import pickle

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def load_inception_v3():
    """Loads the InceptionV3 model with weights pre-trained on ImageNet, excluding the top layers."""
    inception_v3 = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False)
    inception_v3.trainable = False

    # # Reshape the output to be suitable for the Transformer encoder
    # output = inception_v3.output
    # output = tf.keras.layers.Reshape((299, 299, 3))(output)
    #
    # inception_v3 = tf.keras.models.Model(inputs=inception_v3.input, outputs=output)
    return inception_v3


def mobilenet_v3():
    """Loads the MobileNetV3Large model with weights pre-trained on ImageNet, excluding the top layers."""
    mobilenet = tf.keras.applications.MobileNetV3Large(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True)
    mobilenet.trainable = False

    return mobilenet


class Vocabulary:
    def __init__(self, file_path):
        self.vocabulary = self.load_vocabulary(file_path)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.vocabulary)

    def load_vocabulary(self, file_path):
        # Load vocabulary from the file path
        vocabulary = pickle.load(open(file_path, 'rb'))
        return vocabulary

    def tokenize_text(self, text):
        # Tokenize the text
        tokens = text.split()
        token_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        return token_ids

    def lookup_word(self, index):
        # Look up the word by index
        return self.idx2word.get(index, '<unk>')


class CNNEncoder:
    def __init__(self, model):
        self.model = model

    def preprocess_image(self, image):
        # Preprocess image
        image = tf.image.resize(image, (299, 299))
        image = self.model.preprocess_input(image)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training=False):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(query=x, value=x, key=x, attention_mask=None, training=training)
        x = x + attn_output
        return x


class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = keras.layers.Embedding(max_len, embed_dim, input_shape=(None, max_len))

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings
