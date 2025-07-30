from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        if type(features) is models.inception.InceptionOutputs:
            features, _ = features
        features = features.view(features.size(0), -1)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=100):
        self.eval()

        with torch.inference_mode():
            x = self.encoderCNN(image).unsqueeze(0)  # (1, batch_size, embed_size)
            states = None
            result_caption = []

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))  # (batch_size, vocab_size)
                predicted = output.argmax(1)
                predicted_idx = predicted.item()
                result_caption.append(predicted_idx)

                if vocabulary.index_to_word[predicted.item()] == "<EOS>":
                    break

                x = self.decoderRNN.embed(predicted).unsqueeze(0)

            words = [vocabulary.index_to_word.get(idx, "<UNK>") for idx in result_caption]
            return words

    def caption_image_beam_search(self, image, vocabulary, max_length=100, beam_width=3):
        self.eval()
        with torch.inference_mode():
            # Encode image → [1, embed_size]
            features = self.encoderCNN(image)  # [1, embed_size]
            x = features.unsqueeze(0)  # [1, 1, embed_size]

            # Initial hidden state
            sequences = [{
                "tokens": [],
                "log_prob": 0.0,
                "state": None,
                "x": x,
            }]

            for _ in range(max_length):
                all_candidates = []

                for seq in sequences:
                    tokens, log_prob, state, x = seq["tokens"], seq["log_prob"], seq["state"], seq["x"]

                    if tokens and vocabulary.index_to_word[tokens[-1]] == "<EOS>":
                        all_candidates.append(seq)
                        continue

                    hiddens, new_state = self.decoderRNN.lstm(x, state)  # hiddens: [1, 1, hidden]
                    output = self.decoderRNN.linear(hiddens.squeeze(0))  # [1, vocab_size]
                    probs = F.log_softmax(output, dim=1)  # [1, vocab_size]

                    top_log_probs, top_indices = torch.topk(probs, beam_width)

                    for i in range(beam_width):
                        next_token = top_indices[0][i].item()
                        next_log_prob = log_prob + top_log_probs[0][i].item()
                        new_tokens = tokens + [next_token]
                        new_x = self.decoderRNN.embed(top_indices[0][i]).unsqueeze(0).unsqueeze(0)  # [1, 1, embed_size]

                        all_candidates.append({
                            "tokens": new_tokens,
                            "log_prob": next_log_prob,
                            "state": deepcopy(new_state),
                            "x": new_x
                        })

                # Select top `beam_width` sequences
                sequences = sorted(all_candidates, key=lambda c: c["log_prob"], reverse=True)[:beam_width]

            best = sequences[0]["tokens"]
            return [vocabulary.index_to_word.get(idx, "<UNK>") for idx in best]
