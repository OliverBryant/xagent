from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseRerank(ABC):
    """Abstract base class for rerank models."""

    @abstractmethod
    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """
        Rerank documents by relevance to the query.

        Args:
            documents: Candidate documents to rerank
            query: Query to score the documents against

        Returns:
            The documents ordered by descending relevance.
        """
        pass

    def compress_with_scores(
        self,
        documents: Sequence[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Like :meth:`compress` but also returns the relevance score per doc.

        Declared on the base — not only on the concrete providers — so callers
        can reach it through the adapter. Reaching it by unwrapping the
        adapter's inner provider skips usage metering, which is why the RAG
        search pipeline previously recorded no rerank usage at all.

        Returns ``(text, relevance_score)`` tuples ordered by descending
        relevance. The default pairs :meth:`compress` output with a neutral
        score so implementations without native scores still satisfy the
        interface.
        """
        return [(text, 0.0) for text in self.compress(documents, query)]
