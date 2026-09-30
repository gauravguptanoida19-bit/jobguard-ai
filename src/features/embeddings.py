"""
Dense Text Embedding Module for JobGuard AI.
Uses sentence-transformers ('all-MiniLM-L6-v2') with robust disk caching.
"""

import hashlib
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CACHE_PATH = Path("data/processed/embeddings_cache.joblib")


class EmbeddingExtractor:
    """Manages sentence-transformer embeddings with persistent disk cache."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: Optional[Path | str] = DEFAULT_CACHE_PATH,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: dict[str, np.ndarray] = self._load_cache()

    def _load_cache(self) -> dict[str, np.ndarray]:
        if self.cache_path and self.cache_path.exists():
            try:
                data = joblib.load(self.cache_path)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                print(f"[WARNING] Failed to load embeddings cache: {e}. Starting fresh.")
        return {}

    def _save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                joblib.dump(self.cache, self.cache_path, compress=3)
            except Exception as e:
                print(f"[WARNING] Failed to save embeddings cache: {e}")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[INFO] Loading SentenceTransformer '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense 384-dimensional vectors.
        Uses in-memory & disk cache for all previously computed texts.
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        hashes_to_encode = []

        for idx, t in enumerate(texts):
            clean_text = str(t).strip()
            # Fast MD5 hash of text
            h = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
            if h in self.cache:
                results[idx] = self.cache[h]
            else:
                texts_to_encode.append(clean_text if clean_text else "empty")
                indices_to_encode.append(idx)
                hashes_to_encode.append(h)

        if texts_to_encode:
            print(f"[INFO] Encoding {len(texts_to_encode)} new texts with '{self.model_name}'...")
            new_embs = self.model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for idx, h, emb in zip(indices_to_encode, hashes_to_encode, new_embs):
                self.cache[h] = emb
                results[idx] = emb

            self._save_cache()

        return np.vstack(results).astype(np.float32)
