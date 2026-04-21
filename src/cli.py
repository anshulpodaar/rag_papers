"""Command-line interface for the RAG pipeline."""

import argparse
import sys

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


def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Ingest PDFs into the vector store.

    Args:
        args: Parsed CLI arguments.
    """
    from pathlib import Path
    import os

    papers_dir = args.papers_dir or config['papers']['path']
    papers_list = sorted(os.listdir(papers_dir))

    embedder = Embedder()
    store = VectorStore()

    ingested = 0
    for paper in papers_list:
        if Path(paper).suffix not in config['papers']['filetypes']:
            continue

        paper_path = os.path.join(papers_dir, paper)
        logger.info('Processing: %s', paper)

        lines = extract_lines(paper_path)
        if not lines:
            logger.warning('No text extracted from %s', paper)
            continue

        blocks = split_into_sections(lines)
        chunks = chunk_sections(blocks)
        chunks_embedded = embedder.embed(chunks)
        store.upsert(chunks_embedded, source=paper)

        ingested += 1
        print(f'  ✓ {paper} ({len(chunks)} chunks)')

    print(f'\nIngested {ingested} papers. Total chunks: {store.count}')


def cmd_ask(args: argparse.Namespace) -> None:
    """
    Ask a question against the indexed papers.

    Args:
        args: Parsed CLI arguments.
    """
    question = ' '.join(args.question)

    if not question.strip():
        print('Error: Please provide a question.')
        sys.exit(1)

    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    qa = QAEngine(retriever)

    print(f'\nQuestion: {question}\n')
    print('Searching...')

    result = qa.ask_with_sources(
        question,
        top_k=args.top_k,
        context_window=args.context_window,
    )

    transparency = result['transparency']

    # Answer
    print('\n' + '=' * 60)
    print('ANSWER:')
    print('=' * 60)
    print(result['answer'])

    # Confidence & Risk (always show)
    print('\n' + '-' * 60)
    print('CONFIDENCE & RISK:')
    print('-' * 60)
    conf = transparency['confidence']
    risk = transparency['hallucination_risk']
    print(f"  Confidence:        {conf['level'].upper()} ({conf['overall_score']:.2f})")
    print(f"  Hallucination Risk: {risk['level'].upper()} ({risk['score']:.2f})")
    if risk['factors']:
        print(f"  Risk Factors:      {', '.join(risk['factors'])}")

    # Sources
    if args.show_sources:
        print('\n' + '-' * 60)
        print('SOURCES:')
        print('-' * 60)
        for src in transparency['sources']:
            sections = ', '.join(src['sections'][:3])
            if len(src['sections']) > 3:
                sections += f" (+{len(src['sections']) - 3} more)"
            pages = ', '.join(str(p) for p in src['pages'][:5])
            if len(src['pages']) > 5:
                pages += f" (+{len(src['pages']) - 5} more)"
            print(f"  • {src['source']}")
            print(f"    Sections: {sections}")
            print(f"    Pages: {pages}")
            print(f"    Chunks: {src['chunk_count']} | Avg Score: {src['avg_score']:.4f} | Max: {src['max_score']:.4f}")

    # Verbose: chunks and usage
    if args.verbose:
        print('\n' + '-' * 60)
        print('RETRIEVED CHUNKS:')
        print('-' * 60)
        for i, chunk in enumerate(transparency['chunks_retrieved'], 1):
            print(f"  [{i}] Score: {chunk['score']:.4f}")
            print(f"      {chunk['source']} | {chunk['section']} | p.{chunk['page']}")
            print(f"      {chunk['text'][:150]}...")
            print()

        print('-' * 60)
        print(f"Model: {result['model']}")
        print(f"Tokens: {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out")


def cmd_search(args: argparse.Namespace) -> None:
    """
    Search for relevant chunks without generating an answer.

    Args:
        args: Parsed CLI arguments.
    """
    query = ' '.join(args.query)

    if not query.strip():
        print('Error: Please provide a search query.')
        sys.exit(1)

    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)

    print(f'\nSearching for: {query}\n')

    results = retriever.retrieve(
        query,
        top_k=args.top_k,
        source_filter=args.source,
        section_filter=args.section,
    )

    if not results:
        print('No results found.')
        return

    for i, chunk in enumerate(results, 1):
        print(f'[{i}] Score: {chunk["score"]:.4f}')
        print(f'    Source: {chunk["source"]} | {chunk["section"]} | p.{chunk["page"]}')
        print(f'    {chunk["text"][:200]}...\n')


def cmd_list(args: argparse.Namespace) -> None:
    """
    List indexed papers.

    Args:
        args: Parsed CLI arguments.
    """
    store = VectorStore()
    sources = store.list_sources()

    if not sources:
        print('No papers indexed yet. Run: python -m src.cli ingest')
        return

    print(f'\nIndexed papers ({len(sources)}):')
    for source in sources:
        chunks = store.get_by_source(source)
        print(f'  • {source} ({len(chunks)} chunks)')

    print(f'\nTotal chunks: {store.count}')


def cmd_delete(args: argparse.Namespace) -> None:
    """
    Delete a paper from the index.

    Args:
        args: Parsed CLI arguments.
    """
    store = VectorStore()

    if args.source not in store.list_sources():
        print(f'Error: "{args.source}" not found in index.')
        sys.exit(1)

    deleted = store.delete_source(args.source)
    print(f'Deleted {deleted} chunks for "{args.source}"')


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='rag-papers',
        description='RAG pipeline for scientific papers',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest PDFs into vector store')
    ingest_parser.add_argument(
        '--papers-dir',
        type=str,
        default=None,
        help='Path to papers directory (default: from config)',
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', nargs='+', help='Question to ask')
    ask_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of chunks to retrieve')
    ask_parser.add_argument('-c', '--context-window', type=int, default=0, help='Context window size')
    ask_parser.add_argument('-s', '--show-sources', action='store_true', help='Show source citations')
    ask_parser.add_argument('-v', '--verbose', action='store_true', help='Show chunks and token usage')
    ask_parser.set_defaults(func=cmd_ask)

    # search command
    search_parser = subparsers.add_parser('search', help='Search chunks without generating answer')
    search_parser.add_argument('query', nargs='+', help='Search query')
    search_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--source', type=str, default=None, help='Filter by source PDF')
    search_parser.add_argument('--section', type=str, default=None, help='Filter by section')
    search_parser.set_defaults(func=cmd_search)

    # list command
    list_parser = subparsers.add_parser('list', help='List indexed papers')
    list_parser.set_defaults(func=cmd_list)

    # delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a paper from index')
    delete_parser.add_argument('source', type=str, help='Source filename to delete')
    delete_parser.set_defaults(func=cmd_delete)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
