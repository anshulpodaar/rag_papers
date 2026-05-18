"""Experiment runner that ties RAG pipeline, RAGAS evaluation, and tracking."""

from pathlib import Path

import yaml

from src.config import config
from src.embedder import Embedder
from src.evaluation import evaluate_dataset
from src.logger import get_logger
from src.qa_engine import QAEngine
from src.retriever import Retriever
from src.tracking import get_tracker
from src.vector_store import VectorStore

logger = get_logger(__name__)

EVAL_DATASET_PATH = Path('data/eval_dataset.yaml')


def load_eval_dataset(path: Path | None = None) -> list[dict]:
    """
    Load the curated evaluation dataset from YAML.

    Args:
        path: Path to eval dataset YAML. Defaults to data/eval_dataset.yaml.

    Returns:
        List of sample dicts with 'question', 'ground_truth', and 'source'.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    dataset_path = path or EVAL_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f'Evaluation dataset not found at {dataset_path}. '
            f'Create it first with curated Q&A pairs.'
        )

    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)

    return data['samples']


def run_experiment(
    experiment_id: str,
    description: str = '',
    top_k: int = 5,
    context_window: int = 0,
    eval_dataset_path: Path | None = None,
) -> dict:
    """
    Run a full experiment: query pipeline + RAGAS evaluation + tracking.

    For each question in the eval dataset:
    1. Queries the RAG pipeline
    2. Collects answer + retrieved contexts
    3. Evaluates with RAGAS metrics
    4. Logs everything to the configured tracker

    Args:
        experiment_id: Unique identifier for this experiment run.
        description: Human-readable description of what's being tested.
        top_k: Number of chunks to retrieve per question.
        context_window: Context window size for retrieval.
        eval_dataset_path: Override path to eval dataset.

    Returns:
        Dict with 'experiment_id', 'config', 'per_question_results',
        and 'average_scores'.
    """
    # Load eval dataset
    eval_samples = load_eval_dataset(eval_dataset_path)
    logger.info('Loaded %d evaluation samples', len(eval_samples))

    # Initialise pipeline components
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    qa = QAEngine(retriever)

    # Initialise tracker
    tracker = get_tracker()
    experiment_config = {
        'top_k': top_k,
        'context_window': context_window,
        'embedding_model': config['embedding']['model_name'],
        'llm_model': config['claude']['model'],
        'chunk_size': config['chunking']['chunk_size'],
        'chunk_overlap': config['chunking']['chunk_overlap'],
        'n_eval_samples': len(eval_samples),
    }
    tracker.start_experiment(experiment_id, experiment_config, description)

    # Run pipeline on each question and collect results
    ragas_inputs = []
    per_question_results = []

    for i, sample in enumerate(eval_samples, 1):
        question = sample['question']
        ground_truth = sample.get('ground_truth', '')
        source_filter = sample.get('source')

        logger.info('[%d/%d] Evaluating: %s', i, len(eval_samples), question[:50])

        # Query the pipeline
        result = qa.ask(
            question,
            top_k=top_k,
            context_window=context_window,
            source_filter=source_filter,
        )

        # Extract contexts from retrieved chunks
        contexts = [
            chunk['text']
            for chunk in result['transparency']['chunks_retrieved']
        ]

        # Collect for RAGAS batch evaluation
        ragas_inputs.append({
            'question': question,
            'answer': result['answer'],
            'contexts': contexts,
            'ground_truth': ground_truth,
        })

        # Store per-question result
        per_question_results.append({
            'question': question,
            'answer': result['answer'],
            'ground_truth': ground_truth,
            'n_chunks_retrieved': len(contexts),
            'confidence': result['transparency']['confidence'],
            'hallucination_risk': result['transparency']['hallucination_risk'],
            'usage': result['usage'],
        })

    # Run RAGAS evaluation on all samples
    logger.info('Running RAGAS evaluation on %d samples...', len(ragas_inputs))
    avg_scores = evaluate_dataset(ragas_inputs)

    # Log results to tracker
    for i, (question_result, ragas_input) in enumerate(
        zip(per_question_results, ragas_inputs)
    ):
        tracker.log_result(
            question=question_result['question'],
            answer=question_result['answer'],
            metrics={
                'confidence': question_result['confidence']['overall_score'],
                'hallucination_risk': question_result['hallucination_risk']['score'],
                'input_tokens': question_result['usage']['input_tokens'],
                'output_tokens': question_result['usage']['output_tokens'],
            },
            ground_truth=question_result['ground_truth'],
            n_chunks=question_result['n_chunks_retrieved'],
        )

    # Log average RAGAS scores
    for metric_name, score in avg_scores.items():
        tracker.log_metric(metric_name, score)

    # End experiment
    summary = tracker.end_experiment()
    logger.info('Experiment %s complete. Summary: %s', experiment_id, summary)

    return {
        'experiment_id': experiment_id,
        'config': experiment_config,
        'per_question_results': per_question_results,
        'average_scores': avg_scores,
        'summary': summary,
    }


if __name__ == '__main__':
    import sys

    exp_id = sys.argv[1] if len(sys.argv) > 1 else 'baseline_001'
    desc = sys.argv[2] if len(sys.argv) > 2 else 'Baseline experiment run'

    result = run_experiment(experiment_id=exp_id, description=desc)

    print('\n' + '=' * 60)
    print(f'EXPERIMENT: {exp_id}')
    print('=' * 60)
    print(f'\nDescription: {desc}')
    print(f'Samples evaluated: {result["config"]["n_eval_samples"]}')
    print(f'\nAverage RAGAS Scores:')
    for metric, score in result['average_scores'].items():
        print(f'  {metric}: {score:.4f}')
