import collections
import pathlib
import pickle
import re
import string

import einops
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils import *


def flickr8k(path='data/flickr8k'):
    path = pathlib.Path(path)

    captions = (path / 'Flickr8k.token.txt').read_text().splitlines()
    captions = (line.split('\t') for line in captions)
    captions = ((image.split('#')[0], caption) for (image, caption) in captions)

    cap_dict = collections.defaultdict(list)
    for image, cap in captions:
        cap_dict[image].append(cap)

    train_files = (path / 'Flickr_8k.trainImages.txt').read_text().splitlines()
    train_captions = [(str(path / 'Flicker8k_Dataset' / image), cap_dict[image]) for image in train_files]

    test_files = (path / 'Flickr_8k.testImages.txt').read_text().splitlines()
    test_captions = [(str(path / 'Flicker8k_Dataset' / image), cap_dict[image]) for image in test_files]

    train_dataset = tf.data.experimental.from_list(train_captions)
    test_dataset = tf.data.experimental.from_list(test_captions)

    return train_dataset, test_dataset


def ms_coco(path):
    pass


def user_images(path):
    pass


def load_dataset(choice='flickr8k'):
    if choice == 'flickr8k':
        return flickr8k('data/flickr8k')
    elif choice == 'coco':
        return ms_coco('data/coco')
    elif choice == 'users':
        return user_images('data/user_images')


def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    return s


def load_tokenizer(train_dataset):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE,
        standardize=standardize,
        output_sequence_length=MAX_CAPTION_LENGTH,
        ragged=True)

    # Learn the vocabulary
    tokenizer.adapt(train_dataset.map(lambda _, text: text).unbatch().batch(1024))
    return tokenizer



def save_vocabulary(tokenizer):
    pickle.dump(tokenizer.get_vocabulary(), open('data/tokenizer.pkl', 'wb'))


def load_vocabulary(path):
    pickle.load(open(path, 'rb'))


def mappings(tokenizer):
    word_to_index = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True)
    return word_to_index, index_to_word


def match_shapes(images, captions):
    caption_shape = einops.parse_shape(captions, 'b c')
    captions = einops.rearrange(captions, 'b c -> (b c)')
    images = einops.repeat(images, 'b ... -> (b c) ...', c=caption_shape['c'])
    return images, captions


def prepare_txt(images, tokens):
    # tokens = tokenizer(txts)
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return (images, input_tokens), label_tokens


def load_vocab(vocab_path):
    pass
