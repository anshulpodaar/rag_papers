"""Retriever module for semantic search over embedded chunks."""

from src.embedder import Embedder
from src.logger import get_logger
from src.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves semantically similar chunks for a given query.

    Combines the embedder and vector store into a single interface,
    handling query embedding and similarity search in one call.

    Args:
        embedder: Embedder instance for encoding queries.
        vector_store: VectorStore instance for similarity search.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        logger.info('Retriever initialised')

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        section_filter: str | None = None,
        subsection_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            source_filter: Restrict to a specific PDF filename.
            section_filter: Restrict to a specific section name.
            subsection_filter: Restrict to a specific subsection name.

        Returns:
            List of result dicts with keys 'text', 'source', 'section',
            'subsection', 'page', and 'score', sorted by relevance.
        """
        logger.debug('Retrieving for query: %s', query[:50])

        query_embedding = self._embed_query(query)

        results = self._vector_store.query(
            embedding=query_embedding,
            n_results=top_k,
            source_filter=source_filter,
            section_filter=section_filter,
            subsection_filter=subsection_filter,
        )

        logger.info(
            'Retrieved %d chunks (top score: %.4f)',
            len(results),
            results[0]['score'] if results else 0.0,
        )

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1,
        **filters,
    ) -> list[dict]:
        """
        Retrieve chunks with surrounding context.

        Fetches additional chunks before/after each result to provide
        more context for the QA engine.

        Args:
            query: Natural language query string.
            top_k: Number of primary results to return.
            context_window: Number of chunks to include before/after each result.
            **filters: Optional filters (source_filter, section_filter, subsection_filter).

        Returns:
            List of result dicts, potentially expanded with neighboring chunks.
        """
        results = self.retrieve(query, top_k=top_k, **filters)

        if context_window == 0 or not results:
            return results

        expanded = []
        seen_ids = set()

        for result in results:
            source = result['source']
            source_chunks = self._vector_store.get_by_source(source)

            # Find index of current chunk
            current_idx = None
            for i, chunk in enumerate(source_chunks):
                if chunk['text'] == result['text']:
                    current_idx = i
                    break

            if current_idx is None:
                expanded.append(result)
                continue

            # Gather context window
            start = max(0, current_idx - context_window)
            end = min(len(source_chunks), current_idx + context_window + 1)

            for i in range(start, end):
                chunk_id = f"{source}_{i}"

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self._embedder.embed_text(query)

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        # Reuse embedder but for single text
        dummy_chunk = [{'text': query}]
        embedded = self._embedder.embed(dummy_chunk)
        return embedded[0]['embedding']
