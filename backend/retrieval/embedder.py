"""Singleton-style Gemini embedder. One instance per process; safe to share."""
from typing import Sequence

import google.genai as genai


def _build_client(api_key: str):  # broken out for testability
    return genai.Client(api_key=api_key)


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-004", batch_size: int = 32):
        self._client = _build_client(api_key)
        self._model = model
        self._batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        result = self._client.models.embed_content(model=self._model, contents=[text])
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            result = self._client.models.embed_content(model=self._model, contents=batch)
            out.extend(list(e.values) for e in result.embeddings)
        return out
