import torch
import torchvision.transforms as transforms
from PIL import Image

from app.core.config import ALL_DATASETS_DIR


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    image_paths = [
        ALL_DATASETS_DIR / "test_images/dog.jpg",
        ALL_DATASETS_DIR / "test_images/child.jpg",
        ALL_DATASETS_DIR / "test_images/bus.png",
        ALL_DATASETS_DIR / "test_images/boat.png",
        ALL_DATASETS_DIR / "test_images/horse.png",
    ]
    corrects = [
        "Dog on a beach by the ocean",
        "Child holding red frisbee outdoors",
        "Bus driving by parked cars",
        "A small boat in the ocean",
        "A cowboy riding a horse in the desert",
    ]

    for path, correct in zip(image_paths, corrects):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(
            device)
        output = model.caption_image(image, dataset.vocab)
        print(f"CORRECT: {correct}")
        print(f"OUTPUT : {' '.join(output)}\n")

    model.train()


def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
