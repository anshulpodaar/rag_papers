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
        return_context: bool = False,
    ) -> dict:
        """
        Answer a question using retrieved context.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            context_window: Number of neighboring chunks to include (0 = none).
            source_filter: Restrict retrieval to a specific PDF.
            section_filter: Restrict retrieval to a specific section.
            return_context: If True, include retrieved chunks in response.

        Returns:
            Dict with keys:
                - 'answer': Generated response string
                - 'model': Model used
                - 'usage': Token usage dict
                - 'chunks': Retrieved chunks (only if return_context=True)
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
                'chunks': [] if return_context else None,
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

        result = {
            'answer': answer,
            'model': self._model,
            'usage': usage,
        }

        if return_context:
            result['chunks'] = chunks

        return result

    def ask_with_sources(
        self,
        question: str,
        top_k: int = 5,
        **kwargs,
    ) -> dict:
        """
        Answer a question and return structured source citations.

        Convenience method that always returns context and extracts
        unique sources for citation.

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
        """
        result = self.ask(question, top_k=top_k, return_context=True, **kwargs)

        # Extract unique sources
        sources = []
        seen = set()
        for chunk in result.get('chunks', []):
            source_key = (chunk['source'], chunk['section'], chunk['page'])
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    'source': chunk['source'],
                    'section': chunk['section'],
                    'page': chunk['page'],
                    'score': chunk['score'],
                })

        return {
            'answer': result['answer'],
            'sources': sources,
            'model': result['model'],
            'usage': result['usage'],
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
