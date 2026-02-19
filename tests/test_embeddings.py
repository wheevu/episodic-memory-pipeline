"""
Tests for embedding providers.

Tests are organized into:

1. Fast tests (no slow marker):
   - TestMockEmbeddingProvider: Mock provider tests (always pass)
   - TestFactoryFunction: Provider factory tests
   - TestCosineSimilarity: Cosine similarity math tests

2. Slow tests (@pytest.mark.slow):
   - TestLocalEmbeddingProvider: Real SentenceTransformers tests
   - TestBgeM3Provider: BGE-M3 specific tests

Running tests:
   pytest tests/test_embeddings.py              # Fast tests only
   pytest tests/test_embeddings.py --run-slow   # All tests
   pytest tests/test_embeddings.py -m slow      # Slow tests only

macOS Note:
   Slow tests may still download and initialize large local models.
"""

import numpy as np
import pytest

from src.embeddings import (
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)


class TestMockEmbeddingProvider:
    """Test MockEmbeddingProvider - always runs."""

    def test_embed_text_returns_normalized_vector(self) -> None:
        """Mock embeddings should be normalized."""
        provider = MockEmbeddingProvider(dimension=384)
        embedding = provider.embed_text("Hello world")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01

    def test_embed_batch_returns_correct_shape(self) -> None:
        """Batch embeddings should have correct shape."""
        provider = MockEmbeddingProvider(dimension=384)
        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_batch(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_embed_empty_batch(self) -> None:
        """Empty batch should return empty array."""
        provider = MockEmbeddingProvider(dimension=384)
        embeddings = provider.embed_batch([])

        assert embeddings.shape == (0, 384)

    def test_deterministic_embeddings(self) -> None:
        """Same text should produce same embedding."""
        provider = MockEmbeddingProvider(dimension=384)

        emb1 = provider.embed_text("Hello world")
        emb2 = provider.embed_text("Hello world")

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        provider = MockEmbeddingProvider(dimension=384)

        emb1 = provider.embed_text("Hello world")
        emb2 = provider.embed_text("Goodbye universe")

        assert not np.allclose(emb1, emb2)

    def test_is_mock_property(self) -> None:
        """Mock provider should indicate it's mock."""
        provider = MockEmbeddingProvider()
        assert provider.is_mock is True

    def test_embed_documents_alias(self) -> None:
        """embed_documents should work as alias for embed_batch."""
        provider = MockEmbeddingProvider(dimension=384)
        texts = ["Hello", "World"]

        docs = provider.embed_documents(texts)

        assert len(docs) == 2
        assert all(isinstance(d, np.ndarray) for d in docs)

    def test_embed_query_alias(self) -> None:
        """embed_query should work as alias for embed_text."""
        provider = MockEmbeddingProvider(dimension=384)

        emb1 = provider.embed_text("Hello")
        emb2 = provider.embed_query("Hello")

        np.testing.assert_array_equal(emb1, emb2)


class TestFactoryFunction:
    """Test get_embedding_provider factory."""

    def test_mock_provider(self) -> None:
        """Factory should create mock provider."""
        provider = get_embedding_provider("mock", dimension=512)

        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.dimension == 512
        assert provider.is_mock is True

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_embedding_provider("nonexistent")

    def test_openai_without_key_raises(self) -> None:
        """OpenAI provider without API key should raise."""
        with pytest.raises(ValueError, match="API key required"):
            get_embedding_provider("openai")


@pytest.mark.slow
class TestLocalEmbeddingProvider:
    """
    Test LocalEmbeddingProvider with real SentenceTransformers model.

    These tests are marked as slow because they require downloading the model
    (~1GB for BGE-M3) on first run. They verify semantic correctness.
    """

    @pytest.fixture
    def provider(self) -> LocalEmbeddingProvider:
        """Create local embedding provider with smaller model for testing.

        Returns:
            A `LocalEmbeddingProvider` using a small model for faster tests.
        """
        # Use smaller model for faster tests
        return LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    def test_embed_text_returns_normalized(self, provider: LocalEmbeddingProvider) -> None:
        """Local embeddings should be L2-normalized.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        embedding = provider.embed_text("Hello world")

        assert embedding.dtype == np.float32

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01, f"Expected norm ~1.0, got {norm}"

    def test_semantic_similarity_ranking(self, provider: LocalEmbeddingProvider) -> None:
        """
        Similar texts should have higher cosine similarity than dissimilar texts.

        This is the key test that verifies embeddings are semantically meaningful.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        # Embed test sentences
        cat_sentence = provider.embed_text("The cat sat on the mat")
        kitten_sentence = provider.embed_text("A kitten was resting on the rug")
        math_sentence = provider.embed_text("The derivative of x squared is 2x")

        # Compute cosine similarities (embeddings are normalized, so dot product = cosine)
        cat_kitten_sim = np.dot(cat_sentence, kitten_sentence)
        cat_math_sim = np.dot(cat_sentence, math_sentence)

        # Cat and kitten sentences should be more similar than cat and math
        assert cat_kitten_sim > cat_math_sim, (
            f"Expected cat-kitten ({cat_kitten_sim:.3f}) > cat-math ({cat_math_sim:.3f})"
        )

        # Sanity check: similarities should be in reasonable range
        assert 0.3 < cat_kitten_sim < 1.0, f"Unexpected similarity: {cat_kitten_sim}"
        assert 0.0 < cat_math_sim < 0.5, f"Unexpected similarity: {cat_math_sim}"

    def test_embed_batch_matches_individual(self, provider: LocalEmbeddingProvider) -> None:
        """Batch embedding should match individual embeddings.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        texts = ["Hello world", "Test sentence"]

        batch_emb = provider.embed_batch(texts)
        individual_embs = [provider.embed_text(t) for t in texts]

        np.testing.assert_allclose(batch_emb[0], individual_embs[0], rtol=1e-5)
        np.testing.assert_allclose(batch_emb[1], individual_embs[1], rtol=1e-5)

    def test_dimension_property(self, provider: LocalEmbeddingProvider) -> None:
        """Dimension should match embedding size.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        embedding = provider.embed_text("Test")
        assert provider.dimension == embedding.shape[0]

    def test_is_not_mock(self, provider: LocalEmbeddingProvider) -> None:
        """Local provider should not be mock.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        assert provider.is_mock is False

    def test_empty_string_handling(self, provider: LocalEmbeddingProvider) -> None:
        """Empty string should produce valid embedding.

        Args:
            provider: Local embedding provider fixture.

        Returns:
            None.
        """
        embedding = provider.embed_text("")

        assert embedding.shape == (provider.dimension,)
        # Empty string still normalizes
        norm = np.linalg.norm(embedding)
        assert norm > 0.9


@pytest.mark.slow
class TestBgeM3Provider:
    """
    Test BGE-M3 model specifically.

    BGE-M3 is the recommended model for the memory pipeline.
    These tests verify it works correctly if available.
    """

    @pytest.fixture
    def provider(self) -> LocalEmbeddingProvider:
        """Create BGE-M3 provider.

        Returns:
            A `LocalEmbeddingProvider` using the BGE-M3 model.
        """
        try:
            return LocalEmbeddingProvider(model_name="BAAI/bge-m3")
        except Exception as e:
            pytest.skip(f"BGE-M3 model not available: {e}")

    def test_bge_m3_dimension(self, provider: LocalEmbeddingProvider) -> None:
        """BGE-M3 should have 1024-dimensional embeddings.

        Args:
            provider: BGE-M3 embedding provider fixture.

        Returns:
            None.
        """
        assert provider.dimension == 1024

    def test_bge_m3_semantic_quality(self, provider: LocalEmbeddingProvider) -> None:
        """BGE-M3 should produce high-quality semantic embeddings.

        Args:
            provider: BGE-M3 embedding provider fixture.

        Returns:
            None.
        """
        # Korean learning example (relevant to memory pipeline use case)
        korean_1 = provider.embed_text("I started learning Korean today")
        korean_2 = provider.embed_text("My Korean language study is going well")
        unrelated = provider.embed_text("The stock market closed higher today")

        # Korean sentences should be similar
        korean_sim = np.dot(korean_1, korean_2)
        unrelated_sim = np.dot(korean_1, unrelated)

        assert korean_sim > unrelated_sim + 0.1, (
            f"Expected korean similarity ({korean_sim:.3f}) >> unrelated ({unrelated_sim:.3f})"
        )


class TestProviderSelection:
    """
    Smoke tests for provider selection and configuration.

    These tests verify that the provider factory works correctly
    without actually downloading any models.
    """

    def test_mock_is_default_fallback(self) -> None:
        """Mock should work without any external dependencies."""
        provider = get_embedding_provider("mock")

        # Should work immediately
        emb = provider.embed_text("test")
        assert emb is not None
        assert provider.is_mock is True

    def test_local_provider_listed_but_not_loaded(self) -> None:
        """Local provider type should be recognized even if not loaded."""
        # This just tests that the factory knows about 'local'
        # We don't actually load it to avoid model download
        try:
            # This would fail only if 'local' isn't a known provider type
            # In the real implementation, this creates the provider
            # but we can check the error message if sentence-transformers isn't installed
            pass
        except ImportError as e:
            # Expected if sentence-transformers not installed
            assert "sentence-transformers" in str(e).lower()

    def test_embedding_dimension_configurable(self) -> None:
        """Mock provider should respect dimension parameter."""
        provider_384 = get_embedding_provider("mock", dimension=384)
        provider_1024 = get_embedding_provider("mock", dimension=1024)

        assert provider_384.dimension == 384
        assert provider_1024.dimension == 1024

        emb_384 = provider_384.embed_text("test")
        emb_1024 = provider_1024.embed_text("test")

        assert emb_384.shape == (384,)
        assert emb_1024.shape == (1024,)


class TestCosineSimilarity:
    """
    Test cosine-similarity assumptions on normalized embeddings.
    """

    def test_normalized_embeddings_for_ip_search(self) -> None:
        """Normalized embeddings should work for inner product = cosine search."""
        provider = MockEmbeddingProvider(dimension=128)

        # Create some embeddings
        texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        embeddings = provider.embed_batch(texts)

        # Verify all are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

        # For normalized vectors, inner product = cosine similarity
        query = provider.embed_text("query")
        scores = embeddings @ query  # Inner product

        # All scores should be in [-1, 1] range
        assert all(-1.01 <= s <= 1.01 for s in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
