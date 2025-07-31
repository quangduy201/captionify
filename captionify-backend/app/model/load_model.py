import os

import kagglehub
import torch
import pickle

from app import globals
from app.core.config import MODEL_ID, MODEL_CACHE_DIR
from training.model.model import ImageCaptioningModel


def load_model_and_vocab_from_kaggle(force_download=False):
    print("🟡 Downloading model from Kaggle...")
    try:
        if force_download or not os.path.exists(MODEL_CACHE_DIR):
            path = kagglehub.model_download(MODEL_ID, force_download=force_download)
        elif hasattr(globals, "model") and hasattr(globals, "vocab"):
            return True
        else:
            version_numbers = [
                int(d.name) for d in MODEL_CACHE_DIR.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]

            version = max(version_numbers)
            path = MODEL_CACHE_DIR / str(version)

        checkpoint_path = os.path.join(path, "checkpoint.pth.tar")
        vocab_path = os.path.join(path, "vocabulary.pkl")

        print("✅ Model downloaded:", checkpoint_path)

        # Load vocab
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        vocab_size = len(vocab)
        embed_size = 256
        hidden_size = 256
        num_layers = 1

        model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(globals.device)

        checkpoint = torch.load(checkpoint_path, map_location=globals.device)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()

        # Assign to the global variables
        globals.model = model
        globals.vocab = vocab

        print("✅ Model and vocab loaded to RAM")
        return True
    except Exception as e:
        print("❌ Error loading model from Kaggle:", e)
        return False
