import keras
import tensorflow as tf


class ImageCaptioningCNN(tf.keras.Model):
    MAX_CAPTION_LENGTH = 50

    def __init__(self, vocab_size, embedding_dim, num_lstm_units):
        super(ImageCaptioningCNN, self).__init__()

        # Define CNN layers
        self.cnn_model = keras.applications.VGG16(include_top=False, weights='imagenet')
        self.flatten = keras.layers.Flatten()

        # Define LSTM decoder
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = keras.layers.LSTM(num_lstm_units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)  # Output layer for word prediction

    def call(self, inputs):
        # Pass the image through CNN base
        features = self.cnn_model(inputs)
        features = self.flatten(features)

        # LSTM decoder for caption generation
        additional_dimension = tf.expand_dims(features[:, 0], axis=0)
        decoded_caption = self.embedding(tf.zeros_like(additional_dimension))  # Initial input for LSTM
        for i in range(1, self.MAX_CAPTION_LENGTH):  # Loop for caption generation
            predictions = self.dense(self.lstm(decoded_caption))
            predicted_word = tf.argmax(predictions, axis=1)  # Select most likely word
            decoded_caption = tf.concat([decoded_caption, tf.expand_dims(predicted_word, axis=1)], axis=1)
        return decoded_caption  # Return the generated caption

