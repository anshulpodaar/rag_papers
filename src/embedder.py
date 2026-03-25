from sentence_transformers import SentenceTransformer

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Loads a sentence-transformers model and embeds text chunks.

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
        logger.info('Embedding model loaded')

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
        logger.info('Embedding %d chunks', len(texts))

        vectors = self._model.encode(texts, show_progress_bar=True).tolist()

        for chunk, vector in zip(chunks, vectors):
            chunk['embedding'] = vector

        logger.info('Embedding complete — dimensions: %d', len(vectors[0]))
        return chunks