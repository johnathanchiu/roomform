"""Minimal Anthropic client: one image + prompt in, JSON verdict out."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from roomform.agent.errors import ModelError
from roomform.agent.settings import Settings, get_settings

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


class AnthropicClient:
    def __init__(
        self, settings: Settings | None = None, *, client: Any | None = None
    ):
        self.config = settings or get_settings()
        self._client = client

    def judge(self, prompt: str, card_path: Path) -> dict[str, str]:
        """One single-shot call: card image + prompt -> verdict dict."""
        image_png = card_path.read_bytes()
        try:
            response = self._client_or_create().beta.messages.create(
                model=self.config.anthropic_model,
                max_tokens=16000,
                # Server-side fallback: rare classifier refusals re-run
                # on Anthropic's recommended substitute automatically.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _VERDICT_SCHEMA,
                    }
                },
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.standard_b64encode(
                                        image_png
                                    ).decode("ascii"),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise ModelError(f"Anthropic request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            return {"verdict": "reject", "reason": "model refused"}
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError(
                f"Anthropic response was not valid JSON: {text[:200]}"
            ) from exc

    def _client_or_create(self) -> Any:
        if self._client is None:
            api_key = self.config.api_key_for("anthropic")
            if not api_key:
                raise ModelError(
                    "Anthropic credentials required. Set ANTHROPIC_API_KEY."
                )
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
        return self._client


__all__ = ["AnthropicClient"]
