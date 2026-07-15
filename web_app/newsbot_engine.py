"""Free local NLP backend for the NewsBot 2.0 Flask web app.

This file intentionally avoids paid APIs. It trains lightweight scikit-learn
models from the BBC dataset and exposes one simple `analyze_complete` method
for the frontend.
"""

from __future__ import annotations

import re
import zipfile
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


class NewsBotWebEngine:
    """Train and serve the NLP features used by the Flask frontend."""

    def __init__(self, max_articles: int = 1490):
        self.max_articles = max_articles
        self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = None
        self.nlp = None
        self.df = None
        self.vectorizer = None
        self.classifier = None
        self.search_vectorizer = None
        self.search_matrix = None
        self.topic_vectorizer = None
        self.topic_model = None
        self.topic_words = {}
        self.is_ready = False
        self.setup()

    def setup(self) -> None:
        """Download small NLTK resources, load spaCy, dataset, and models."""
        for resource in ["stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

        self.stop_words = set(stopwords.words("english"))
        self.sia = SentimentIntensityAnalyzer()
        self.nlp = self._load_spacy_model()
        self.df = self._load_dataset()
        self._train_classifier()
        self._train_search_index()
        self._train_topic_model()
        self.is_ready = True

    def _load_spacy_model(self):
        """Load the small English spaCy model, installing it if needed."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")

    def _find_dataset_zip(self) -> Path:
        """Find learn-ai-bbc.zip in common local project/download locations."""
        current = Path(__file__).resolve()
        candidates = [
            current.parent / "data" / "learn-ai-bbc.zip",
            current.parent.parent / "learn-ai-bbc.zip",
            current.parent.parent / "data" / "raw" / "learn-ai-bbc.zip",
            Path("learn-ai-bbc.zip"),
            Path.home() / "Downloads" / "learn-ai-bbc.zip",
            Path(r"C:\Users\Admin\Downloads\learn-ai-bbc.zip"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find learn-ai-bbc.zip. Put it in web_app/data/ or the project root."
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Load and prepare the BBC News Classification dataset."""
        data_dir = Path(__file__).resolve().parent / "data" / "extracted"
        data_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self._find_dataset_zip()) as archive:
            archive.extractall(data_dir)

        csv_path = next(data_dir.rglob("BBC News Train.csv"))
        raw_df = pd.read_csv(csv_path)
        df = raw_df.rename(
            columns={"ArticleId": "article_id", "Text": "content", "Category": "category"}
        )[["article_id", "content", "category"]].copy()

        df["content"] = df["content"].fillna("").astype(str).str.strip()
        df["category"] = df["category"].fillna("").astype(str).str.strip()
        df = df[(df["content"].str.len() >= 200) & (df["category"].str.len() > 0)].copy()
        if len(df) > self.max_articles:
            df = df.sample(self.max_articles, random_state=42).copy()

        df["title"] = (
            df["content"]
            .str.split(r"\s{2,}|\n", n=1)
            .str[0]
            .str.split()
            .str[:16]
            .str.join(" ")
            .str.title()
        )
        df["processed"] = df["content"].apply(self.preprocess_text)
        return df.reset_index(drop=True)

    def clean_text(self, text: str) -> str:
        """Clean raw text before tokenization."""
        text = "" if text is None else str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_text(self, text: str) -> str:
        """Clean, tokenize, remove stop words, and lemmatize text."""
        tokens = re.findall(r"[a-z]+", self.clean_text(text))
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def _extra_features(self, texts) -> np.ndarray:
        """Create sentiment and length features."""
        rows = []
        for text in texts:
            text = "" if text is None else str(text)
            scores = self.sia.polarity_scores(text[:5000])
            rows.append(
                [
                    (scores["compound"] + 1) / 2,
                    scores["pos"],
                    scores["neu"],
                    scores["neg"],
                    min(len(text.split()) / 1000, 1),
                    min(len(text) / 7000, 1),
                ]
            )
        return np.array(rows, dtype=float)

    def _feature_matrix(self, texts, fit: bool = False):
        """Combine TF-IDF with lightweight numeric features."""
        processed = [self.preprocess_text(text) for text in texts]
        if fit:
            text_matrix = self.vectorizer.fit_transform(processed)
        else:
            text_matrix = self.vectorizer.transform(processed)
        return hstack([text_matrix, csr_matrix(self._extra_features(texts))]).tocsr().copy()

    def _train_classifier(self) -> None:
        """Train a multiclass classifier with confidence scores."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                C=4.0,
                class_weight="balanced",
                random_state=42,
            )
        )
        X = self._feature_matrix(self.df["content"], fit=True)
        y = self.df["category"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.classifier.fit(X_train, y_train)
        self.model_accuracy = float(self.classifier.score(X_test, y_test))

    def _train_search_index(self) -> None:
        """Create a TF-IDF similarity index for semantic-style search."""
        self.search_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
        self.search_matrix = self.search_vectorizer.fit_transform(self.df["processed"])

    def _train_topic_model(self) -> None:
        """Train an NMF topic model for topic discovery."""
        self.topic_vectorizer = TfidfVectorizer(max_features=3500, min_df=2, max_df=0.85)
        topic_matrix = self.topic_vectorizer.fit_transform(self.df["processed"])
        self.topic_model = NMF(n_components=6, init="nndsvda", random_state=42, max_iter=400)
        self.topic_model.fit(topic_matrix)

        feature_names = self.topic_vectorizer.get_feature_names_out()
        for topic_id, weights in enumerate(self.topic_model.components_):
            top_indices = weights.argsort()[-8:][::-1]
            self.topic_words[topic_id] = [feature_names[index] for index in top_indices]

    def classify(self, text: str) -> dict:
        """Classify article text and return probabilities."""
        X = self._feature_matrix([text], fit=False)
        probabilities = self.classifier.predict_proba(X)[0]
        best_index = int(np.argmax(probabilities))
        classes = self.classifier.classes_
        confidence = float(probabilities[best_index])
        return {
            "category": str(classes[best_index]),
            "confidence": confidence,
            "probabilities": {
                str(label): round(float(prob), 4) for label, prob in zip(classes, probabilities)
            },
        }

    def sentiment(self, text: str) -> dict:
        """Analyze article sentiment with VADER."""
        scores = self.sia.polarity_scores(text)
        if scores["compound"] >= 0.05:
            label = "positive"
        elif scores["compound"] <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, **scores}

    def summarize(self, text: str, n_sentences: int = 3) -> str:
        """Create a simple extractive summary from high-scoring sentences."""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(sentence.strip().split()) >= 6
        ]
        if len(sentences) <= n_sentences:
            return " ".join(sentences)

        tokens = self.preprocess_text(text).split()
        word_scores = Counter(tokens)
        max_score = max(word_scores.values()) if word_scores else 1
        word_scores = {word: score / max_score for word, score in word_scores.items()}

        scored = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = self.preprocess_text(sentence).split()
            score = sum(word_scores.get(token, 0) for token in sentence_tokens) / max(len(sentence_tokens), 1)
            scored.append((index, score, sentence))
        best = sorted(scored, key=lambda item: item[1], reverse=True)[:n_sentences]
        return " ".join(sentence for _, _, sentence in sorted(best, key=lambda item: item[0]))

    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from article text."""
        doc = self.nlp(text[:5000])
        entities = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT"}:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_) or ent.label_,
                    }
                )
        return entities[:20]

    def topic(self, text: str) -> dict:
        """Assign article text to a discovered topic."""
        processed = self.preprocess_text(text)
        vector = self.topic_vectorizer.transform([processed])
        distribution = self.topic_model.transform(vector)[0]
        topic_id = int(np.argmax(distribution))
        return {
            "topic_id": topic_id,
            "strength": round(float(distribution[topic_id]), 4),
            "top_words": self.topic_words.get(topic_id, []),
        }

    def similar_articles(self, text: str, top_n: int = 3) -> list[dict]:
        """Find similar BBC articles using TF-IDF cosine similarity."""
        processed = self.preprocess_text(text)
        query_vector = self.search_vectorizer.transform([processed])
        scores = cosine_similarity(query_vector, self.search_matrix).ravel()
        top_indices = scores.argsort()[-top_n:][::-1]
        results = []
        for index in top_indices:
            row = self.df.iloc[int(index)]
            results.append(
                {
                    "title": row["title"],
                    "category": row["category"],
                    "similarity": round(float(scores[index]), 4),
                }
            )
        return results

    def keywords(self, text: str, top_n: int = 10) -> list[str]:
        """Return top processed keywords."""
        return [word for word, _ in Counter(self.preprocess_text(text).split()).most_common(top_n)]

    def analyze_complete(self, text: str) -> dict:
        """Run the complete web-app analysis pipeline."""
        if not text or len(text.strip()) < 20:
            return {"success": False, "error": "Please provide at least 20 characters of article text."}

        text = text.strip()
        classification = self.classify(text)
        sentiment = self.sentiment(text)
        topic = self.topic(text)
        entities = self.extract_entities(text)
        summary = self.summarize(text)
        keywords = self.keywords(text)
        similar = self.similar_articles(text)

        insights = [
            f"Predicted category is {classification['category']} with {classification['confidence']:.1%} confidence.",
            f"Sentiment is {sentiment['label']} with compound score {sentiment['compound']:.3f}.",
            f"Dominant topic uses terms: {', '.join(topic['top_words'][:5])}.",
        ]
        if entities:
            insights.append("Key entities: " + ", ".join(sorted({entity["text"] for entity in entities[:6]})) + ".")

        return {
            "success": True,
            "category": classification["category"],
            "confidence": round(classification["confidence"], 4),
            "probabilities": classification["probabilities"],
            "sentiment": sentiment,
            "summary": summary,
            "entities": entities,
            "keywords": keywords,
            "topic": topic,
            "similar_articles": similar,
            "statistics": {
                "word_count": len(text.split()),
                "character_count": len(text),
                "model_accuracy": round(self.model_accuracy, 4),
            },
            "insights": insights,
            "processed_preview": self.clean_text(text)[:300] + ("..." if len(text) > 300 else ""),
        }
