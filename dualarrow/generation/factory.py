from __future__ import annotations

from dualarrow.config import GeneratorConfig
from dualarrow.errors import ConfigurationError
from dualarrow.generation.base import AnswerGenerator
from dualarrow.generation.extractive import ExtractiveGenerator
from dualarrow.generation.openai_compatible import OpenAICompatibleGenerator


def create_generator(config: GeneratorConfig) -> AnswerGenerator:
    if config.provider == "extractive":
        return ExtractiveGenerator()
    if config.provider == "openai_compatible":
        if not config.base_url:
            raise ConfigurationError("generation.base_url is required")
        return OpenAICompatibleGenerator(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported generator provider: {config.provider}")
