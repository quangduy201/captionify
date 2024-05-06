import glob
import os

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.sequence import pad_sequences


def load_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normalize pixel values

    return img_array


def load_captions(captions_file):
    # Open the captions file
    with open(captions_file, 'r') as f:
        captions = f.readlines()

    # Strip every caption
    captions = [cap.strip() for cap in captions]

    return captions


def create_caption_map(captions, tokenizer):
    # Tokenize captions for vocabulary building
    tokenized_captions = tokenizer.texts_to_sequences(captions)

    # Pad sequences to a fixed length (optional)
    max_length = max(len(seq) for seq in tokenized_captions)
    padded_captions = pad_sequences(tokenized_captions, maxlen=max_length, padding='post')

    # Create a word-to-index map for vocabulary
    word_index_map = tokenizer.word_index

    return padded_captions, word_index_map


def create_word_to_index_map(tokenizer):
    # Create a word-to-index map for vocabulary
    word_index_map = tokenizer.word_index

    return word_index_map


def load_data(image_folder, captions_file, tokenizer):
    # Load image paths from the folder
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

    # Load captions from captions file
    captions = load_captions(captions_file)

    # Ensure image and caption count match
    if len(image_paths) != len(captions):
        raise ValueError('Number of images and captions do not match.')

    # Load the preprocess images and captions
    image_features = [load_image(path) for path in image_paths]
    padded_captions, word_index_map = create_caption_map(captions, tokenizer)

    return image_features, padded_captions, word_index_map
