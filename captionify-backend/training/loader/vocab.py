from typing import List

from app.utils.spacy_utils import get_spacy_model


class Vocabulary:
    def __init__(self, frequency_threshold: int):
        self.index_to_word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_to_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.index_to_word)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in get_spacy_model().tokenizer(text)]

    def build_vocabulary(self, sentence_list: List[str]) -> None:
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

    def numericalize(self, text: str) -> List[int]:
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.word_to_index.get(token, self.word_to_index["<UNK>"])
            for token in tokenized_text
        ]

    def __getstate__(self):
        return {
            "index_to_word": self.index_to_word,
            "word_to_index": self.word_to_index,
            "frequency_threshold": self.frequency_threshold,
        }

    def __setstate__(self, state):
        self.index_to_word = state["index_to_word"]
        self.word_to_index = state["word_to_index"]
        self.frequency_threshold = state["frequency_threshold"]
