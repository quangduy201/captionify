import pathlib

import spacy
import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to an index
# 2. We need to set up a PyTorch dataset to load the data
# 3. Setup padding of every batch (all examples should be of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_english = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, frequency_threshold):
        self.index_to_word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_to_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.index_to_word)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_english.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        index = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.frequency_threshold:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.word_to_index[token] if token in self.word_to_index else self.word_to_index["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = pathlib.Path(root_dir)
        self.data = self.load_annotations(captions_file)
        self.transform = transform

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary([caption for _, caption in self.data])

    def load_annotations(self, annotations_file):
        data = (self.root_dir / annotations_file).read_text().splitlines()
        data = [line.split('\t') for line in data]
        data = [(image_file.split('#')[0], caption) for image_file, caption in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_file, caption = self.data[index]
        img = Image.open((self.root_dir/"flickr8k_images"/image_file)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.word_to_index["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word_to_index["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return images, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.word_to_index["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), ]
    )

    loader, dataset = get_loader(
        "../../data/flickr8k",
        "flickr8k_annotations/Flickr8k.token.txt",
        transform=transform
    )

    for idx, (images, captions) in enumerate(loader):
        print(f"{idx}: {images.shape} - {captions.shape}")
