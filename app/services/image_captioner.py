from typing import BinaryIO

import torch
import torchvision.transforms as transforms
from PIL import Image

import app
from app import globals
from app.model.load_model import load_model_and_vocab_from_kaggle
from app.utils.text import normalize_caption


def preprocess_image(file: BinaryIO) -> torch.Tensor:
    file.seek(0)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(globals.device)
    return image_tensor


def postprocess_caption(tokens: list[str]) -> str:
    filtered_tokens = []
    for word in tokens:
        if word in ("<SOS>", "<EOS>", "<PAD>"):
            continue
        if word == "<UNK>":
            filtered_tokens.append("[unknown]")
        else:
            filtered_tokens.append(word)

    if not filtered_tokens:
        return ""

    caption = normalize_caption(filtered_tokens)
    return caption


def generate_caption(file: BinaryIO) -> str:
    image_tensor = preprocess_image(file)

    if not hasattr(app.globals, "model") or not hasattr(app.globals, "vocab"):
        load_model_and_vocab_from_kaggle()

    tokens = globals.model.caption_image_beam_search(image_tensor, globals.vocab)
    caption = postprocess_caption(tokens)
    return caption
