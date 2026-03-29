"""Main entry point for the RAG pipeline."""

import os
from pathlib import Path

from src.chunker import chunk_sections
from src.config import config
from src.embedder import Embedder
from src.extractor import extract_lines
from src.logger import get_logger
from src.qa_engine import QAEngine
from src.retriever import Retriever
from src.section_detector import split_into_sections
from src.vector_store import VectorStore

logger = get_logger(__name__)


def ingest_papers() -> None:
    """
    Ingest all PDFs from the configured papers directory.

    Extracts text, chunks, embeds, and stores in vector database.
    """
    papers_dir_path = config['papers']['path']
    papers_filetypes = config['papers']['filetypes']
    papers_list = sorted(os.listdir(papers_dir_path))

    embedder = Embedder()
    store = VectorStore()

    for paper in papers_list:
        if Path(paper).suffix not in papers_filetypes:
            continue

        paper_path = os.path.join(papers_dir_path, paper)
        lines = extract_lines(paper_path)

        if not lines:
            logger.warning('No text extracted from %s', paper)
            continue

        blocks = split_into_sections(lines)
        chunks = chunk_sections(blocks)

        logger.info(
            'Paper: %s | Lines: %d | Blocks: %d | Chunks: %d',
            paper, len(lines), len(blocks), len(chunks),
        )

        chunks_embedded = embedder.embed(chunks)
        store.upsert(chunks_embedded, source=paper)

    logger.info('Ingestion complete. Total chunks in store: %d', store.count)


def query_papers(question: str, top_k: int = 5) -> dict:
    """
    Query the RAG pipeline with a question.

    Args:
        question: Natural language question.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with 'answer', 'sources', 'model', and 'usage'.
    """
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    qa = QAEngine(retriever)

    result = qa.ask_with_sources(question, top_k=top_k)
    return result


if __name__ == '__main__':
    # Ingest papers (run once or when papers change)
    # ingest_papers()

    # Example query
    question = 'What is the attention mechanism and how does it work?'
    logger.info('Question: %s', question)

    result = query_papers(question)

    print('\n' + '=' * 60)
    print('ANSWER:')
    print('=' * 60)
    print(result['answer'])

    print('\n' + '-' * 60)
    print('SOURCES:')
    print('-' * 60)
    for src in result['sources']:
        print(f"  • {src['source']} | {src['section']} | p.{src['page']} (score: {src['score']})")

    print('\n' + '-' * 60)
    print(f"Model: {result['model']}")
    print(f"Tokens: {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out")
