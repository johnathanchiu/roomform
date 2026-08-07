from __future__ import annotations

from roomform.agent.provider.anthropic import AnthropicClient
from roomform.agent.provider.claude_cli import ClaudeCLIClient
from roomform.agent.provider.openai import OpenAIClient
from roomform.agent.settings import Settings

Client = AnthropicClient | ClaudeCLIClient | OpenAIClient


def create_client(settings: Settings) -> Client:
    if settings.provider == "openai":
        return OpenAIClient(settings)
    if settings.provider == "claude-cli":
        return ClaudeCLIClient()
    return AnthropicClient(settings)


__all__ = [
    "AnthropicClient",
    "ClaudeCLIClient",
    "OpenAIClient",
    "create_client",
]
