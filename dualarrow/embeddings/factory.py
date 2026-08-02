from __future__ import annotations

from dualarrow.config import EmbeddingConfig
from dualarrow.embeddings.base import EmbeddingProvider
from dualarrow.embeddings.hash import HashEmbeddingProvider
from dualarrow.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from dualarrow.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from dualarrow.errors import ConfigurationError


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "hash":
        return HashEmbeddingProvider(config.dimensions)
    if config.provider == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(config.model, config.dimensions)
    if config.provider == "openai_compatible":
        if not config.base_url:
            raise ConfigurationError("embedding.base_url is required")
        return OpenAICompatibleEmbeddingProvider(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            dimensions=config.dimensions,
        )
    raise ConfigurationError(f"Unsupported embedding provider: {config.provider}")
