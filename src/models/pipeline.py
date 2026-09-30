"""
Unified JobGuard Production ML Pipeline.
Bundles feature engineering, dense embeddings, classifier, and decision threshold
into a single deployable artifact with joblib persistence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.features.engineering import JobFeatureExtractor
from src.features.embeddings import EmbeddingExtractor
from src.data.cleaner import clean_job_postings


class JobGuardPipeline(BaseEstimator, ClassifierMixin):
    """
    End-to-end production pipeline for JobGuard AI.
    Supports 'tfidf', 'hybrid', and 'tabular' feature representations.
    """

    def __init__(
        self,
        classifier: Any = None,
        feature_mode: str = "hybrid",
        threshold: float = 0.50,
        model_name: str = "JobGuard-Pipeline",
    ):
        self.classifier = classifier
        self.feature_mode = feature_mode
        self.threshold = threshold
        self.model_name = model_name
        self.feature_extractor = JobFeatureExtractor()
        self.tfidf_vectorizer: Optional[Any] = None
        self.embedding_extractor: Optional[EmbeddingExtractor] = None
        if feature_mode == "hybrid":
            self.embedding_extractor = EmbeddingExtractor()
        self.feature_names_: List[str] = []

    def _prepare_dataframe(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Ensure input is a cleaned DataFrame with required schema."""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # If combined_text is missing, apply cleaner
        if "combined_text" not in df.columns:
            df = clean_job_postings(df, drop_duplicates=False, create_combined_text=True)

        return df

    def transform_features(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Tuple[Any, List[str]]:
        """
        Transform raw posting(s) into feature matrix and feature name list.
        """
        df = self._prepare_dataframe(X)

        if self.feature_mode == "tfidf":
            if self.tfidf_vectorizer is None:
                raise ValueError("tfidf_vectorizer must be fitted before transforming features in tfidf mode.")
            texts = df["combined_text"].fillna("").tolist()
            X_mat = self.tfidf_vectorizer.transform(texts)
            return X_mat, self.feature_names_

        tabular_feats = self.feature_extractor.transform(df)
        tabular_names = list(tabular_feats.columns)

        if self.feature_mode == "hybrid" and self.embedding_extractor is not None:
            texts = df["combined_text"].fillna("").tolist()
            dense_embs = self.embedding_extractor.encode_texts(texts, show_progress=False)
            emb_names = [f"emb_{i}" for i in range(dense_embs.shape[1])]
            all_matrix = np.hstack([tabular_feats.values, dense_embs]).astype(np.float32)
            all_names = tabular_names + emb_names
            return all_matrix, all_names

        return tabular_feats.values.astype(np.float32), tabular_names

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit feature extractor and downstream classifier."""
        df = self._prepare_dataframe(X)
        self.feature_extractor.fit(df, y)
        X_mat, self.feature_names_ = self.transform_features(df)
        print(f"[INFO] Fitting classifier on feature matrix of shape {X_mat.shape}...")
        self.classifier.fit(X_mat, y)
        return self

    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """Return class probabilities [P(0), P(1)]."""
        X_mat, _ = self.transform_features(X)
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_mat)
            if probs.ndim == 2:
                return probs
            return np.column_stack([1.0 - probs, probs])
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X_mat)
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - probs, probs])
        else:
            raise NotImplementedError("Classifier does not support probability estimation.")

    def predict(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary class labels based on decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def save(self, filepath: Union[str, Path]):
        """Persist pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid saving large unpicklable objects; embedding extractor cache is on disk
        joblib.dump(self, path, compress=3)
        print(f"[SUCCESS] Pipeline saved to '{path}'.")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "JobGuardPipeline":
        """Load persisted pipeline from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at '{path.resolve()}'. Please run training pipeline first.")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is of type {type(pipeline)}, expected {cls.__name__}.")
        return pipeline
