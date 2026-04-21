"""QA engine for answering questions using retrieved context and Claude."""

from anthropic import Anthropic

from src.config import config
from src.logger import get_logger
from src.retriever import Retriever

logger = get_logger(__name__)

# Default prompt template — designed for A/B testing swaps
DEFAULT_SYSTEM_PROMPT = """You are a precise research assistant answering questions based on scientific papers.

Instructions:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information, say so explicitly
- Cite the source (paper name, section, page) when making claims
- Be concise but complete
- Use technical language appropriate for the source material"""

DEFAULT_USER_TEMPLATE = """Context from research papers:
{context}

---

Question: {question}

Answer based on the context above:"""


class QAEngine:
    """
    Answers questions by retrieving relevant chunks and generating responses.

    Combines retrieval and generation into a single interface. Designed for
    easy A/B testing of different prompts, models, and retrieval strategies.

    Args:
        retriever: Retriever instance for fetching relevant chunks.
        model: Claude model identifier. Defaults to config value.
        max_tokens: Maximum response tokens. Defaults to config value.
        system_prompt: System prompt for Claude. Defaults to DEFAULT_SYSTEM_PROMPT.
        user_template: User message template with {context} and {question} placeholders.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._model = model or config['claude']['model']
        self._max_tokens = max_tokens or config['claude']['max_tokens']
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._user_template = user_template or DEFAULT_USER_TEMPLATE

        self._client = Anthropic()
        logger.info('QAEngine initialised with model: %s', self._model)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        context_window: int = 0,
        source_filter: str | None = None,
        section_filter: str | None = None,
    ) -> dict:
        """
        Answer a question with full transparency metadata.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            context_window: Number of neighboring chunks to include (0 = none).
            source_filter: Restrict retrieval to a specific PDF.
            section_filter: Restrict retrieval to a specific section.

        Returns:
            Dict with keys:
                - 'answer': Generated response string
                - 'model': Model used
                - 'usage': Token usage dict
                - 'transparency': Confidence and source transparency metadata
        """
        # Retrieve relevant chunks
        if context_window > 0:
            chunks = self._retriever.retrieve_with_context(
                query=question,
                top_k=top_k,
                context_window=context_window,
                source_filter=source_filter,
                section_filter=section_filter,
            )
        else:
            chunks = self._retriever.retrieve(
                query=question,
                top_k=top_k,
                source_filter=source_filter,
                section_filter=section_filter,
            )

        if not chunks:
            logger.warning('No chunks retrieved for question: %s', question[:50])
            return {
                'answer': 'No relevant context found to answer this question.',
                'model': self._model,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
                'transparency': self._build_empty_transparency(),
            }

        # Format context for prompt
        context = self._format_context(chunks)

        # Build user message
        user_message = self._user_template.format(
            context=context,
            question=question,
        )

        # Call Claude
        logger.debug('Sending request to Claude (%d chunks in context)', len(chunks))
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[{'role': 'user', 'content': user_message}],
        )

        answer = response.content[0].text
        usage = {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens,
        }

        logger.info(
            'Generated answer (%d input tokens, %d output tokens)',
            usage['input_tokens'],
            usage['output_tokens'],
        )

        # Build transparency metadata
        transparency = self._build_transparency(chunks, answer)

        return {
            'answer': answer,
            'model': self._model,
            'usage': usage,
            'transparency': transparency,
        }

    def ask_with_sources(
        self,
        question: str,
        top_k: int = 5,
        **kwargs,
    ) -> dict:
        """
        Answer a question and return structured source citations.

        Convenience method that extracts unique sources for citation.
        Now includes full transparency metadata.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            **kwargs: Additional arguments passed to ask().

        Returns:
            Dict with keys:
                - 'answer': Generated response string
                - 'sources': List of unique source citations
                - 'model': Model used
                - 'usage': Token usage dict
                - 'transparency': Full transparency metadata
        """
        result = self.ask(question, top_k=top_k, **kwargs)

        # Extract unique sources from transparency
        sources = result['transparency']['sources']

        return {
            'answer': result['answer'],
            'sources': sources,
            'model': result['model'],
            'usage': result['usage'],
            'transparency': result['transparency'],
        }

    def _build_transparency(self, chunks: list[dict], answer: str) -> dict:
        """
        Build transparency metadata from retrieved chunks and answer.

        Args:
            chunks: List of retrieved chunk dicts.
            answer: Generated answer string.

        Returns:
            Dict containing:
                - chunks_retrieved: Full list of chunks with metadata
                - sources: Unique sources with aggregated scores
                - confidence: Overall confidence metrics
                - hallucination_risk: Risk assessment for multi-source answers
        """
        # Build chunks retrieved
        chunks_retrieved = [
            {
                'text': chunk['text'],
                'source': chunk['source'],
                'section': chunk['section'],
                'subsection': chunk.get('subsection', ''),
                'page': chunk['page'],
                'score': chunk['score'],
            }
            for chunk in chunks
        ]

        # Aggregate sources
        source_scores: dict[str, dict] = {}
        for chunk in chunks:
            source = chunk['source']
            if source not in source_scores:
                source_scores[source] = {
                    'source': source,
                    'sections': set(),
                    'pages': set(),
                    'chunk_count': 0,
                    'scores': [],
                }
            source_scores[source]['sections'].add(chunk['section'])
            source_scores[source]['pages'].add(chunk['page'])
            source_scores[source]['chunk_count'] += 1
            source_scores[source]['scores'].append(chunk['score'])

        sources = []
        for source_data in source_scores.values():
            scores = source_data['scores']
            sources.append({
                'source': source_data['source'],
                'sections': sorted(source_data['sections']),
                'pages': sorted(source_data['pages']),
                'chunk_count': source_data['chunk_count'],
                'avg_score': round(sum(scores) / len(scores), 4),
                'max_score': round(max(scores), 4),
                'min_score': round(min(scores), 4),
            })

        # Sort by avg_score descending
        sources.sort(key=lambda x: x['avg_score'], reverse=True)

        # Calculate confidence metrics
        confidence = self._calculate_confidence(chunks)

        # Calculate hallucination risk
        hallucination_risk = self._calculate_hallucination_risk(chunks, sources)

        return {
            'chunks_retrieved': chunks_retrieved,
            'sources': sources,
            'confidence': confidence,
            'hallucination_risk': hallucination_risk,
        }

    def _calculate_confidence(self, chunks: list[dict]) -> dict:
        """
        Calculate confidence metrics based on retrieval quality.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Dict with confidence metrics.
        """
        if not chunks:
            return {
                'overall_score': 0.0,
                'top_chunk_score': 0.0,
                'score_spread': 0.0,
                'avg_score': 0.0,
                'level': 'none',
            }

        scores = [c['score'] for c in chunks]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        score_spread = top_score - min(scores)

        # Overall confidence: weighted combination
        # High top score + low spread = high confidence (chunks agree)
        # High top score + high spread = medium confidence (mixed relevance)
        overall = (top_score * 0.6) + (avg_score * 0.3) + ((1 - score_spread) * 0.1)

        # Determine confidence level
        if overall >= 0.75:
            level = 'high'
        elif overall >= 0.50:
            level = 'medium'
        elif overall >= 0.25:
            level = 'low'
        else:
            level = 'very_low'

        return {
            'overall_score': round(overall, 4),
            'top_chunk_score': round(top_score, 4),
            'avg_score': round(avg_score, 4),
            'score_spread': round(score_spread, 4),
            'level': level,
        }

    def _calculate_hallucination_risk(
        self,
        chunks: list[dict],
        sources: list[dict],
    ) -> dict:
        """
        Calculate hallucination risk based on source diversity and score distribution.

        Higher risk when:
        - Multiple sources with conflicting information potential
        - Low retrieval scores (weak grounding)
        - High score variance (inconsistent relevance)

        Args:
            chunks: List of retrieved chunks.
            sources: Aggregated source information.

        Returns:
            Dict with hallucination risk assessment.
        """
        if not chunks:
            return {
                'score': 1.0,
                'level': 'high',
                'factors': ['no_context_retrieved'],
            }

        factors = []
        risk_score = 0.0

        # Factor 1: Number of unique sources
        num_sources = len(sources)
        if num_sources > 2:
            risk_score += 0.2
            factors.append(f'multi_source ({num_sources} documents)')
        elif num_sources == 2:
            risk_score += 0.1
            factors.append('dual_source')

        # Factor 2: Low average retrieval score
        avg_score = sum(c['score'] for c in chunks) / len(chunks)
        if avg_score < 0.4:
            risk_score += 0.3
            factors.append(f'weak_grounding (avg_score={avg_score:.2f})')
        elif avg_score < 0.6:
            risk_score += 0.15
            factors.append(f'moderate_grounding (avg_score={avg_score:.2f})')

        # Factor 3: High score variance (inconsistent relevance)
        scores = [c['score'] for c in chunks]
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            if variance > 0.04:  # std > 0.2
                risk_score += 0.2
                factors.append(f'high_variance (var={variance:.3f})')
            elif variance > 0.01:  # std > 0.1
                risk_score += 0.1
                factors.append(f'moderate_variance (var={variance:.3f})')

        # Factor 4: Top score too low
        top_score = max(scores)
        if top_score < 0.5:
            risk_score += 0.25
            factors.append(f'low_top_score ({top_score:.2f})')

        # Factor 5: Cross-section retrieval (potential topic mixing)
        all_sections = set()
        for chunk in chunks:
            all_sections.add(chunk['section'])
        if len(all_sections) > 3:
            risk_score += 0.1
            factors.append(f'topic_spread ({len(all_sections)} sections)')

        # Normalize and cap at 1.0
        risk_score = min(risk_score, 1.0)

        # Determine risk level
        if risk_score >= 0.6:
            level = 'high'
        elif risk_score >= 0.35:
            level = 'medium'
        elif risk_score >= 0.15:
            level = 'low'
        else:
            level = 'minimal'

        if not factors:
            factors.append('well_grounded')

        return {
            'score': round(risk_score, 4),
            'level': level,
            'factors': factors,
        }

    def _build_empty_transparency(self) -> dict:
        """Build empty transparency metadata when no chunks retrieved."""
        return {
            'chunks_retrieved': [],
            'sources': [],
            'confidence': {
                'overall_score': 0.0,
                'top_chunk_score': 0.0,
                'avg_score': 0.0,
                'score_spread': 0.0,
                'level': 'none',
            },
            'hallucination_risk': {
                'score': 1.0,
                'level': 'high',
                'factors': ['no_context_retrieved'],
            },
        }

    def _format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Args:
            chunks: List of chunk dicts from retriever.

        Returns:
            Formatted context string with source citations.
        """
        formatted_chunks = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            section = chunk.get('section', 'Unknown')
            page = chunk.get('page', '?')
            text = chunk.get('text', '')

            formatted_chunks.append(
                f'[{i}] Source: {source} | Section: {section} | Page: {page}\n{text}'
            )

        return '\n\n'.join(formatted_chunks)
