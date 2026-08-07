"""Agent settings: provider selection + credentials (ROOMFORM_ env
prefix, `.env` aware). The harness is a single-shot VLM judge; there is
no run loop to configure."""

from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PROVIDER = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-5.5-2026-04-23"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ROOMFORM_",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["anthropic", "openai", "claude-cli"] = Field(
        default=DEFAULT_PROVIDER, validation_alias="ROOMFORM_PROVIDER"
    )
    anthropic_model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        validation_alias="ROOMFORM_ANTHROPIC_MODEL",
    )
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    openai_model: str = Field(
        default=DEFAULT_OPENAI_MODEL, validation_alias="ROOMFORM_MODEL"
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )

    def api_key_for(self, provider: str) -> str:
        key = (
            self.anthropic_api_key
            if provider == "anthropic"
            else self.openai_api_key
        )
        return (key or "").strip()


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
