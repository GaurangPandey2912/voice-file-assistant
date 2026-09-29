from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("voice_assistant.search.semantic")

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticEmbedder:
    """Thin, lazily-loaded wrapper around a sentence-transformers model.

    Loading the model (and its PyTorch dependency) is relatively expensive and
    requires a one-time download, so nothing is imported until the first call
    to encode() — a session that never needs semantic search never pays that
    cost, and if the package/model isn't available the engine transparently
    falls back to keyword/fuzzy matching instead of crashing.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._available = True

    def _ensure_model(self):
        if self._model is not None or not self._available:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading semantic embedding model '{self._model_name}'...")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Semantic embedding model loaded")
        except Exception as e:
            logger.warning(
                f"Semantic search unavailable ({e}); falling back to "
                "keyword/fuzzy matching only."
            )
            self._available = False
        return self._model

    @property
    def is_available(self) -> bool:
        if self._model is None and self._available:
            self._ensure_model()
        return self._available

    def encode(self, texts: list[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        model = self._ensure_model()
        if model is None:
            return None
        try:
            return model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"Embedding encode failed: {e}")
            return None

    def encode_one(self, text: str) -> Optional[np.ndarray]:
        result = self.encode([text])
        if result is None or len(result) == 0:
            return None
        return result[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Both vectors are assumed pre-normalized (encode() does this), so a
        plain dot product is the cosine similarity."""
        return float(np.dot(a, b))
