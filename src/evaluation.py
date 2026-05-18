"""RAGAS evaluation module for scoring RAG pipeline quality."""

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.config import config, get_api_key
from src.logger import get_logger

logger = get_logger(__name__)

METRIC_NAMES = [
    'faithfulness',
    'answer_relevancy',
    'context_precision',
    'context_recall',
]


# ── LLM setup for RAGAS judge ────────────────────────────────────────────────


def _get_evaluator_llm() -> LangchainLLMWrapper:
    """
    Get the LLM used by RAGAS as the evaluation judge.

    Uses Claude via langchain-anthropic. Falls back to config model.

    Returns:
        LangchainLLMWrapper wrapping a ChatAnthropic instance.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    from langchain_anthropic import ChatAnthropic

    api_key = get_api_key('anthropic')
    if not api_key:
        raise ValueError(
            'ANTHROPIC_API_KEY not set. Required for RAGAS evaluation.'
        )

    model = config['claude']['model']
    llm = ChatAnthropic(model=model, api_key=api_key)
    return LangchainLLMWrapper(llm)


def _get_metrics(evaluator_llm: LangchainLLMWrapper, include_recall: bool = True) -> list:
    """
    Create RAGAS metric instances with the evaluator LLM.

    Args:
        evaluator_llm: The LLM wrapper to use for evaluation.
        include_recall: Whether to include context_recall (requires ground_truth).

    Returns:
        List of metric instances.
    """
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ]
    if include_recall:
        metrics.append(ContextRecall(llm=evaluator_llm))
    return metrics


# ── Core evaluation function ─────────────────────────────────────────────────


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a single question-answer pair using RAGAS metrics.

    Args:
        question: The user question.
        answer: The generated answer from the RAG pipeline.
        contexts: List of retrieved context strings used to generate the answer.
        ground_truth: Expected correct answer (required for context_recall).

    Returns:
        Dict mapping metric names to scores (0.0 to 1.0).
    """
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth or '',
    )

    dataset = EvaluationDataset(samples=[sample])
    evaluator_llm = _get_evaluator_llm()

    # Select metrics based on available data
    metrics_to_run = _get_metrics(evaluator_llm, include_recall=bool(ground_truth))

    logger.info('Running RAGAS evaluation (%d metrics)', len(metrics_to_run))

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=evaluator_llm,
    )

    scores = result.to_pandas().iloc[0].to_dict()

    # Clean up: keep only metric scores, drop input columns
    metric_scores = {}
    for name in METRIC_NAMES:
        if name in scores:
            metric_scores[name] = round(float(scores[name]), 4)

    logger.info('RAGAS scores: %s', metric_scores)
    return metric_scores


def evaluate_dataset(
    samples: list[dict],
) -> dict[str, float]:
    """
    Evaluate a full dataset of question-answer pairs.

    Args:
        samples: List of dicts with keys:
            - 'question' (str): The user question
            - 'answer' (str): Generated answer
            - 'contexts' (list[str]): Retrieved context strings
            - 'ground_truth' (str, optional): Expected answer

    Returns:
        Dict with average scores across all samples for each metric.
    """
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s['question'],
                response=s['answer'],
                retrieved_contexts=s['contexts'],
                reference=s.get('ground_truth', ''),
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)
    evaluator_llm = _get_evaluator_llm()

    # Use all metrics if any sample has ground_truth
    has_ground_truth = any(s.get('ground_truth') for s in samples)
    metrics_to_run = _get_metrics(evaluator_llm, include_recall=has_ground_truth)

    logger.info(
        'Running RAGAS evaluation on %d samples (%d metrics)',
        len(samples),
        len(metrics_to_run),
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=evaluator_llm,
    )

    df = result.to_pandas()

    # Compute averages
    avg_scores = {}
    for name in METRIC_NAMES:
        if name in df.columns:
            avg_scores[name] = round(float(df[name].mean()), 4)

    logger.info('Average RAGAS scores: %s', avg_scores)
    return avg_scores
