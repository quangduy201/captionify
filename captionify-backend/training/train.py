import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from app.core.config import CHECKPOINT_PATH, VOCAB_PATH, FLICKR8K_DIR, ANNOTATION_FILE
from training.loader.get_loader import get_loader
from training.model.model import ImageCaptioningModel
from training.model.utils import load_checkpoint, print_examples, save_checkpoint


def train():
    load_model = use_existing_vocab = False
    save_model = False
    train_cnn = True

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if use_existing_vocab and os.path.exists(VOCAB_PATH):
        print(f"Loading vocabulary from {VOCAB_PATH}")
        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = None
        print("No existing vocabulary found. Building new vocabulary...")

    train_loader, dataset = get_loader(
        dataset_name="flickr8k",
        root_folder=FLICKR8K_DIR,
        annotation_file=ANNOTATION_FILE,
        transform=transform,
        num_workers=2,
        vocab=vocab,
    )

    if vocab is None:
        print("Saving newly built vocabulary...")
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(dataset.vocab, f)  # type: ignore

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # For tensorboard
    writer = SummaryWriter("training/logs/runs/flickr")
    step = 0

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss, optimizer
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word_to_index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only fine-tune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        param.requires_grad = train_cnn if "fc" not in name else True

    if load_model and os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        step = load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=device), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        # Uncomment the line below to see a couple of test cases
        print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, CHECKPOINT_PATH)

            with open(VOCAB_PATH, "wb") as f:
                pickle.dump(dataset.vocab, f)  # type: ignore

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train()
