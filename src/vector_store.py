"""ChromaDB vector store for RAG pipeline."""

import chromadb

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Manages a persistent ChromaDB collection for storing and retrieving
    embedded chunks.

    Args:
        db_path: Path to the ChromaDB persistence directory.
            Defaults to config value.
        collection_name: Name of the collection to use.
            Defaults to config value.
    """

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._db_path = db_path or config['vector_store']['db_path']
        self._collection_name = collection_name or config['vector_store']['collection_name']

        logger.info('Initialising ChromaDB at %s', self._db_path)
        self._client = chromadb.PersistentClient(path=self._db_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={'hnsw:space': 'cosine'},  # Explicit distance metric
        )
        logger.info(
            'Collection: %s | Existing chunks: %d',
            self._collection_name,
            self.count,
        )

    def upsert(self, chunks: list[dict], source: str) -> int:
        """
        Store chunks in ChromaDB, replacing any existing chunks for
        this source to make re-ingestion idempotent.

        Args:
            chunks: List of chunk dicts with keys 'text', 'embedding',
                'section', 'subsection', and 'page'.
            source: PDF filename used as a namespace for chunk IDs.

        Returns:
            Number of chunks stored.
        """
        self._delete_by_source(source)

        ids = [f'{source}_chunk_{i}' for i in range(len(chunks))]
        documents = [c['text'] for c in chunks]
        embeddings = [c['embedding'] for c in chunks]
        metadatas = [
            {
                'source': source,
                'section': c['section'],
                'subsection': c['subsection'] or '',
                'page': c['page'],
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info('Stored %d chunks for %s', len(chunks), source)
        return len(chunks)

    def query(
        self,
        embedding: list[float],
        n_results: int | None = None,
        source_filter: str | None = None,
        section_filter: str | None = None,
        subsection_filter: str | None = None,
    ) -> list[dict]:
        """
        Find the most semantically similar chunks to a query embedding.

        Args:
            embedding: Query vector from the embedder.
            n_results: Number of results to return. Defaults to config value.
            source_filter: Restrict to a specific PDF filename.
            section_filter: Restrict to a specific section name.
            subsection_filter: Restrict to a specific subsection name.

        Returns:
            List of result dicts with keys 'text', 'source', 'section',
            'subsection', 'page', and 'score'.
        """
        n = n_results or config['vector_store']['n_results']
        where = self._build_filter(source_filter, section_filter, subsection_filter)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=['documents', 'metadatas', 'distances'],
        )

        return self._parse_results(results)

    def get_by_source(self, source: str) -> list[dict]:
        """
        Retrieve all chunks for a given source.

        Args:
            source: PDF filename to retrieve.

        Returns:
            List of chunk dicts with 'text' and metadata.
        """
        results = self._collection.get(
            where={'source': source},
            include=['documents', 'metadatas'],
        )
        return [
            {
                'text': doc,
                'source': meta.get('source', ''),
                'section': meta.get('section', ''),
                'subsection': meta.get('subsection', ''),
                'page': meta.get('page', 0),
            }
            for doc, meta in zip(results['documents'], results['metadatas'])
        ]

    def delete_source(self, source: str) -> int:
        """
        Delete all chunks for a given source.

        Args:
            source: PDF filename to delete.

        Returns:
            Number of chunks deleted.
        """
        return self._delete_by_source(source)

    def list_sources(self) -> list[str]:
        """
        Return sorted list of all unique source filenames in the collection.

        Returns:
            Sorted list of source filenames.
        """
        all_meta = self._collection.get(include=['metadatas'])
        sources = {m['source'] for m in all_meta['metadatas']}
        return sorted(sources)

    @property
    def count(self) -> int:
        """Total number of chunks currently stored."""
        return self._collection.count()

    # ── Private helpers ───────────────────────────────────────────────

    def _delete_by_source(self, source: str) -> int:
        """Delete existing chunks for a source, return count deleted."""
        existing = self._collection.get(where={'source': source})
        if existing['ids']:
            self._collection.delete(ids=existing['ids'])
            logger.debug('Deleted %d existing chunks for %s', len(existing['ids']), source)
            return len(existing['ids'])
        return 0

    @staticmethod
    def _build_filter(
        source: str | None,
        section: str | None,
        subsection: str | None,
    ) -> dict | None:
        """Build ChromaDB where clause from optional filters."""
        conditions = {}
        if source:
            conditions['source'] = source
        if section:
            conditions['section'] = section
        if subsection:
            conditions['subsection'] = subsection

        if len(conditions) > 1:
            return {'$and': [{k: v} for k, v in conditions.items()]}
        return conditions or None

    @staticmethod
    def _parse_results(raw: dict) -> list[dict]:
        """Convert ChromaDB response to clean result dicts."""
        docs = raw.get('documents', [[]])[0]
        metas = raw.get('metadatas', [[]])[0]
        distances = raw.get('distances', [[]])[0]

        return [
            {
                'text': doc,
                'source': meta.get('source', ''),
                'section': meta.get('section', ''),
                'subsection': meta.get('subsection', ''),
                'page': meta.get('page', 0),
                'score': round(1 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]
