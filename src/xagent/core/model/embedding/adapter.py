import logging
from typing import List, Optional, Union

import requests

from ...retry import create_retry_wrapper
from ..model import EmbeddingModelConfig
from .base import BaseEmbedding
from .dashscope import DashScopeEmbedding
from .openai import OpenAIEmbedding
from .xinference import XinferenceEmbedding

logger = logging.getLogger(__name__)


def _estimate_tokens(text: Union[str, List[str]]) -> int:
    """Rough token estimate for embedding usage (~4 chars/token).

    Embedding providers here don't return a usage payload, so we approximate
    from character count. Good enough for cost-tracking granularity.
    """
    texts = [text] if isinstance(text, str) else list(text)
    chars = sum(len(t) for t in texts if isinstance(t, str))
    return chars // 4


def retry_on(e: Exception) -> bool:
    ERRORS = requests.exceptions.Timeout

    if isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code
        return status_code == 429 or 500 <= status_code < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def create_embedding_adapter(model_config: EmbeddingModelConfig) -> BaseEmbedding:
    """
    Creates a custom BaseEmbedding instance from an EmbeddingModelConfig.
    """
    embedding = EmbeddingModelAdapter(model_config)

    return create_retry_wrapper(
        embedding,
        BaseEmbedding,  # type: ignore[type-abstract]
        retry_methods={"encode"},
        max_retries=model_config.max_retries,
        retry_on=retry_on,
    )


class EmbeddingModelAdapter(BaseEmbedding):
    """Adapter that makes the new embedding interface compatible with existing EmbeddingModel configs."""

    def __init__(self, model_config: EmbeddingModelConfig):
        self.model_config = model_config
        self._embedding_model = self._create_embedding_model()

    def _create_embedding_model(self) -> BaseEmbedding:
        """Create the actual embedding model from configuration."""
        # Normalize provider name: map variants to canonical names
        # Note: 'openai_embedding' is a legacy value that should be treated as 'openai'
        provider = self.model_config.model_provider.lower().strip()
        if provider in ("openai", "openai_embedding", "openai-compatible"):
            return OpenAIEmbedding(
                model=self.model_config.model_name,
                api_key=self.model_config.api_key,
                base_url=self.model_config.base_url,
                dimension=self.model_config.dimension,
            )
        elif provider == "dashscope":
            return DashScopeEmbedding(
                model=self.model_config.model_name,
                api_key=self.model_config.api_key,
                base_url=self.model_config.base_url,
                dimension=self.model_config.dimension,
                instruct=self.model_config.instruct,
            )
        elif provider == "xinference":
            return XinferenceEmbedding(
                model=self.model_config.model_name,
                base_url=self.model_config.base_url,
                api_key=self.model_config.api_key,
                dimension=self.model_config.dimension,
            )
        else:
            raise ValueError(
                f"Unsupported model provider: {self.model_config.model_provider}"
            )

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """Encode text using the underlying embedding model."""
        result = self._embedding_model.encode(text, dimension, instruct)
        try:
            # Lazy import: this module is on the model package's init critical
            # path, and importing ..chat at top level would circularly re-enter
            # ..model before ChatModelConfig is defined.
            from ..chat.token_context import add_media_usage

            count = 1 if isinstance(text, str) else len(text)
            add_media_usage(
                unit="requests",
                quantity=count,
                model=self.model_config.model_name,
                call_type="embedding",
                input_tokens=_estimate_tokens(text),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to record embedding usage: %s", e)
        return result

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._embedding_model.get_dimension()

    @property
    def abilities(self) -> List[str]:
        """Get the model abilities."""
        return self._embedding_model.abilities
