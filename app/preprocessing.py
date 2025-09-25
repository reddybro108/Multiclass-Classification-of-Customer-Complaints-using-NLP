# src/preprocessing.py
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

class Preprocessor:
    def __init__(self, tfidf_max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        self.glove_embeddings = self.load_glove("embeddings/glove.6B.100d.txt")
        self.embedding_dim = 100

    def load_glove(self, file_path):
        embeddings = {}
        with open(file_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                embeddings[word] = vector
        return embeddings

    def clean_text(self, text: str) -> str:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return " ".join(tokens)

    def text_to_glove(self, text: str) -> np.ndarray:
        tokens = text.split()
        vectors = [self.glove_embeddings.get(token, np.zeros(self.embedding_dim)) for token in tokens]
        return np.mean(vectors, axis=0)

    def fit_transform_tfidf(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform_tfidf(self, texts):
        return self.vectorizer.transform(texts).toarray()
