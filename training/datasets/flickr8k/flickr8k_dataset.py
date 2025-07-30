import pathlib

import torch
from torch.utils.data import Dataset
from PIL import Image

from training.loader.vocab import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, image_subdir="flickr8k_images", vocab=None):
        self.root_dir = pathlib.Path(root_dir)
        self.captions_path = (
            self.root_dir / captions_file if isinstance(captions_file, str) else pathlib.Path(captions_file)
        )
        self.image_subdir = image_subdir
        self.data = self.load_annotations()
        self.transform = transform

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary([caption for _, caption in self.data])

    def load_annotations(self):
        lines = self.captions_path.read_text(encoding="utf-8").splitlines()
        data = [line.split('\t') for line in lines]
        return [(image_file.split('#')[0], caption) for image_file, caption in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_file, caption = self.data[index]
        img_path = self.root_dir / self.image_subdir / image_file

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.word_to_index["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word_to_index["<EOS>"])

        return img, torch.tensor(numericalized_caption)
