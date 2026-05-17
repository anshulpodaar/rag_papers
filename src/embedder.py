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

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """
        Embed one or more text strings.

        Accepts a single string or a list of strings. Always returns
        a list of embedding vectors (one per input text).

        Args:
            text: A single text string or list of text strings to embed.

        Returns:
            List of embedding vectors (list of list of floats).

        Raises:
            TypeError: If input is not a string or list.
            ValueError: If input is empty or contains empty/non-string items.
        """
        if not isinstance(text, (str, list)):
            raise TypeError(f'Expected str or list[str], got {type(text).__name__}.')

        if isinstance(text, str):
            if not text.strip():
                raise ValueError('Cannot embed empty string.')
            text = [text]

        if not text:
            raise ValueError('Cannot embed empty input.')

        if not all(isinstance(t, str) and t.strip() for t in text):
            raise ValueError('All items must be non-empty strings.')

        show_progress = len(text) > 1
        logger.info('Embedding %d text(s)', len(text))
        vectors = self._model.encode(text, show_progress_bar=show_progress)
        logger.info('Embedding complete')
        return vectors.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunks, adding an 'embedding' key to each.

        Pipeline convenience method that extracts text from chunk dicts,
        embeds in batch, and attaches vectors back to the dicts.

        Args:
            chunks: List of chunk dicts with at least a 'text' key,
                as returned by chunker.chunk_sections().

        Returns:
            The same list with an 'embedding' (list[float]) key added
            to each chunk.
        """
        texts = [c['text'] for c in chunks]
        vectors = self.embed(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk['embedding'] = vector

        return chunks
