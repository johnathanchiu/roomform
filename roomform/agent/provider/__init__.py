from __future__ import annotations

from roomform.agent.protocols import Model
from roomform.agent.provider.anthropic import AnthropicModel
from roomform.agent.provider.anthropic import (
    context_window_tokens_for as anthropic_context_window_tokens_for,
)
from roomform.agent.provider.gemini import GeminiModel
from roomform.agent.provider.gemini import (
    context_window_tokens_for as gemini_context_window_tokens_for,
)
from roomform.agent.provider.openai import OpenAIModel
from roomform.agent.provider.openai import (
    context_window_tokens_for as openai_context_window_tokens_for,
)
from roomform.agent.provider.openrouter import OpenRouterModel
from roomform.agent.settings import Settings


def create_model(settings: Settings) -> Model:
    if settings.provider == "openrouter":
        return OpenRouterModel(settings)
    if settings.provider == "anthropic":
        return AnthropicModel(settings)
    if settings.provider == "gemini":
        return GeminiModel(settings)
    return OpenAIModel(settings)


def context_window_tokens_for(model: str) -> int | None:
    return (
        openai_context_window_tokens_for(model)
        or gemini_context_window_tokens_for(model)
        or anthropic_context_window_tokens_for(model)
    )


__all__ = [
    "AnthropicModel",
    "GeminiModel",
    "OpenAIModel",
    "OpenRouterModel",
    "context_window_tokens_for",
    "create_model",
]
