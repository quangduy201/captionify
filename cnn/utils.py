import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)

    # Normalize pixel values (e.g., divide by 255)
    img_array = img_array / 255.0
    return img_array


def extract_features(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    features = model.predict(np.expand_dims(preprocessed_image, axis=0))
    return features


def load_trained_model(model_path):
    model = load_model(model_path)
    return model


def generate_caption(image_path, model, word_to_index_map):
    # Preprocessing the image
    preprocessed_image = preprocess_image(image_path)
    image_feature = np.expand_dims(preprocessed_image, axis=0)

    # Generate caption using the LSTM decoder
    caption = model.predict(image_feature)[0]
    predicted_caption = ''
    for word_index in caption:
        if word_index != 0:  # Skip padding (0)
            predicted_caption += word_to_index_map[word_index] + ' '
    predicted_caption = predicted_caption.strip()

    return predicted_caption
