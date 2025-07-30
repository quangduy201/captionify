import spacy

_spacy_model = None

def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            print("🟡 Downloading en_core_web_sm...")
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_model = spacy.load("en_core_web_sm")
    return _spacy_model
