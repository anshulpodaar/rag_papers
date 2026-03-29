"""Embedding module using sentence-transformers."""

from sentence_transformers import SentenceTransformer

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Loads a sentence-transformers model and embeds text.

    Initialises the model once and reuses it across calls — loading
    a transformer model is expensive and should never happen per-chunk.

    Args:
        model_name: Sentence-transformers model identifier.
            Defaults to config value.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config['embedding']['model_name']
        logger.info('Loading embedding model: %s', self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info('Embedding model loaded — dimensions: %d', self._dimensions)

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for this model."""
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        vector = self._model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        logger.info('Embedding %d texts', len(texts))
        vectors = self._model.encode(texts, show_progress_bar=True)
        logger.info('Embedding complete')
        return vectors.tolist()

    def embed(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunks, adding an 'embedding' key to each.

        Args:
            chunks: List of chunk dicts with at least a 'text' key,
                as returned by chunker.chunk_sections().

        Returns:
            The same list with an 'embedding' (list[float]) key added
            to each chunk.
        """
        texts = [c['text'] for c in chunks]
        vectors = self.embed_batch(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk['embedding'] = vector

        return chunks
