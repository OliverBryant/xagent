import logging
from collections.abc import Sequence

import requests

from ...retry import create_retry_wrapper
from ..model import RerankModelConfig
from .base import BaseRerank

logger = logging.getLogger(__name__)


def retry_on(e: Exception) -> bool:
    ERRORS = requests.exceptions.Timeout

    if isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code
        return status_code == 429 or 500 <= status_code < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def _create_rerank_model(model_config: RerankModelConfig) -> BaseRerank:
    """Create the underlying rerank model based on ``model_provider``."""
    provider = (model_config.model_provider or "dashscope").lower()

    if provider == "xinference":
        from .xinference import XinferenceRerank

        return XinferenceRerank(
            model=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            top_n=model_config.top_n,
            timeout=model_config.timeout,
        )

    # Default: DashScope-compatible rerank endpoint
    from .dashscope import DashscopeRerank

    return DashscopeRerank(
        model=model_config.model_name,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        top_n=model_config.top_n,
        instruct=model_config.instruct,
    )


def create_rerank_adapter(model_config: RerankModelConfig) -> BaseRerank:
    """
    Creates a custom BaseRerank instance from a RerankModelConfig with retry logic.
    """
    return create_retry_wrapper(
        RerankModelAdapter(model_config),
        BaseRerank,  # type: ignore[type-abstract]
        retry_methods={"compress"},
        max_retries=model_config.max_retries,
        retry_on=retry_on,
    )


class RerankModelAdapter(BaseRerank):
    """Adapter that makes the new rerank interface compatible with existing RerankModel configs."""

    def __init__(self, model_config: RerankModelConfig):
        self.model_config = model_config
        self._rerank_model = _create_rerank_model(model_config)

    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """Rerank documents using the underlying rerank model."""
        result = self._rerank_model.compress(documents, query)
        try:
            # Lazy import to avoid a circular import via the model package init.
            from ..chat.token_context import add_media_usage

            doc_chars = (
                sum(len(d) for d in documents if isinstance(d, str)) if documents else 0
            )
            query_len = len(query) if isinstance(query, str) else 0
            add_media_usage(
                unit="requests",
                quantity=1,
                model=self.model_config.model_name,
                call_type="rerank",
                input_tokens=(doc_chars + query_len) // 4,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to record rerank usage: %s", e)
        return result
