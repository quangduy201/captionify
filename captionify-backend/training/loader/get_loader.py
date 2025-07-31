import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from training.datasets.flickr8k import FlickrDataset
from training.loader.vocab import Vocabulary


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
    dataset_name: str,
    root_folder: str,
    annotation_file: str,
    transform,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    vocab: Vocabulary = None
):
    if dataset_name.lower() == "flickr8k":
        dataset = FlickrDataset(root_folder, annotation_file, transform=transform, vocab=vocab)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported yet.")

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
