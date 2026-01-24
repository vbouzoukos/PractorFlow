"""
Text summarization tool.

Provides extractive summarization for long text content.
Useful for processing large documents within context limits.
"""

import re
from typing import List

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class TextSummarizerTool(BaseTool):
    """
    Text summarization tool using extractive summarization.

    Extracts the most important sentences from text based on
    word frequency and sentence scoring. No external dependencies required.
    """

    def __init__(
        self,
        default_num_sentences: int = 5,
        max_input_length: int = 50000,
    ):
        """
        Initialize text summarizer tool.

        Args:
            default_num_sentences: Default number of sentences to extract.
            max_input_length: Maximum input text length in characters.
        """
        self._default_num_sentences = default_num_sentences
        self._max_input_length = max_input_length

    @property
    def name(self) -> str:
        return "summarize_text"

    @property
    def description(self) -> str:
        return (
            "Summarize long text content by extracting the most important sentences. "
            "Use this tool when you need to condense large documents or text into "
            "a shorter, more manageable summary."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="The text content to summarize",
                required=True,
            ),
            ToolParameter(
                name="num_sentences",
                type="integer",
                description=f"Number of sentences to extract (default: {self._default_num_sentences})",
                required=False,
                default=self._default_num_sentences,
            ),
            ToolParameter(
                name="preserve_order",
                type="boolean",
                description="Keep sentences in original order (default: true)",
                required=False,
                default=True,
            ),
        ]

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:
                cleaned.append(sent)

        return cleaned

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Extract words from text.

        Args:
            text: Input text.

        Returns:
            List of lowercase words.
        """
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return words

    def _get_stopwords(self) -> set:
        """Get common English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'dare', 'ought', 'used', 'this', 'that', 'these', 'those', 'it',
            'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'who', 'whom', 'which', 'what',
            'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'then', 'once', 'any', 'into', 'about',
            'after', 'before', 'above', 'below', 'between', 'under', 'over',
        }

    def _calculate_word_frequencies(self, text: str) -> dict:
        """
        Calculate word frequencies excluding stopwords.

        Args:
            text: Input text.

        Returns:
            Dictionary of word frequencies.
        """
        words = self._tokenize_words(text)
        stopwords = self._get_stopwords()

        frequencies = {}
        for word in words:
            if word not in stopwords:
                frequencies[word] = frequencies.get(word, 0) + 1

        if frequencies:
            max_freq = max(frequencies.values())
            for word in frequencies:
                frequencies[word] = frequencies[word] / max_freq

        return frequencies

    def _score_sentence(self, sentence: str, word_frequencies: dict) -> float:
        """
        Score a sentence based on word frequencies.

        Args:
            sentence: Sentence to score.
            word_frequencies: Dictionary of normalized word frequencies.

        Returns:
            Sentence score.
        """
        words = self._tokenize_words(sentence)
        if not words:
            return 0.0

        score = sum(word_frequencies.get(word, 0) for word in words)
        return score / len(words)

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute text summarization.

        Args:
            text: Text content to summarize.
            num_sentences: Number of sentences to extract.
            preserve_order: Whether to keep original sentence order.

        Returns:
            ToolResult with summary or error.
        """
        text = kwargs.get("text")
        num_sentences = kwargs.get("num_sentences", self._default_num_sentences)
        preserve_order = kwargs.get("preserve_order", True)

        if not text:
            return ToolResult(success=False, error="Text parameter is required")

        if len(text) > self._max_input_length:
            return ToolResult(
                success=False,
                error=f"Text exceeds maximum length of {self._max_input_length} characters"
            )

        try:
            logger.debug(f"[TextSummarizer] Summarizing {len(text)} characters")

            sentences = self._tokenize_sentences(text)

            if len(sentences) <= num_sentences:
                return ToolResult(
                    success=True,
                    data=text,
                    metadata={
                        "original_sentences": len(sentences),
                        "extracted_sentences": len(sentences),
                        "compression_ratio": 1.0,
                        "note": "Text already shorter than requested summary length",
                    },
                )

            word_frequencies = self._calculate_word_frequencies(text)

            scored_sentences = []
            for idx, sentence in enumerate(sentences):
                score = self._score_sentence(sentence, word_frequencies)
                scored_sentences.append((idx, sentence, score))

            sorted_by_score = sorted(scored_sentences, key=lambda x: x[2], reverse=True)
            top_sentences = sorted_by_score[:num_sentences]

            if preserve_order:
                top_sentences = sorted(top_sentences, key=lambda x: x[0])

            summary = " ".join(sent[1] for sent in top_sentences)

            compression_ratio = len(summary) / len(text) if text else 0

            logger.debug(
                f"[TextSummarizer] Extracted {num_sentences} sentences, "
                f"compression ratio: {compression_ratio:.2f}"
            )

            return ToolResult(
                success=True,
                data=summary,
                metadata={
                    "original_sentences": len(sentences),
                    "extracted_sentences": num_sentences,
                    "original_length": len(text),
                    "summary_length": len(summary),
                    "compression_ratio": round(compression_ratio, 3),
                },
            )

        except Exception as e:
            logger.error(f"[TextSummarizer] Error: {e}")
            return ToolResult(success=False, error=f"Summarization failed: {str(e)}")