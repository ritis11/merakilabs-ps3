from unittest.mock import MagicMock
import pytest
from backend.retrieval.embedder import GeminiEmbedder


@pytest.fixture
def embedder(monkeypatch):
    fake_client = MagicMock()

    def fake_embed(model, contents):
        # Return as many fake 4-d embeddings as inputs.
        result = MagicMock()
        result.embeddings = [
            MagicMock(values=[float(i), float(i) + 0.1, 0.0, 1.0])
            for i, _ in enumerate(contents)
        ]
        return result

    fake_client.models.embed_content.side_effect = fake_embed
    monkeypatch.setattr(
        "backend.retrieval.embedder._build_client",
        lambda key: fake_client,
    )
    return GeminiEmbedder(api_key="test", model="text-embedding-004", batch_size=2)


def test_embed_documents_batches(embedder):
    out = embedder.embed_documents(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert all(len(v) == 4 for v in out)


def test_embed_query_returns_single_vector(embedder):
    v = embedder.embed_query("what was revenue?")
    assert isinstance(v, list)
    assert len(v) == 4
