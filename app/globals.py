import torch

from training.loader.vocab import Vocabulary
from training.model.model import ImageCaptioningModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: ImageCaptioningModel
vocab: Vocabulary
