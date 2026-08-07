"""Minimal OpenAI client: one image + prompt in, JSON verdict out."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from roomform.agent.errors import ModelError
from roomform.agent.settings import Settings, get_settings


class OpenAIClient:
    def __init__(
        self, settings: Settings | None = None, *, client: Any | None = None
    ):
        self.config = settings or get_settings()
        self._client = client

    def judge(self, prompt: str, card_path: Path) -> dict[str, str]:
        """One single-shot call: card image + prompt -> verdict dict."""
        data = base64.standard_b64encode(card_path.read_bytes()).decode(
            "ascii"
        )
        try:
            response = self._client_or_create().chat.completions.create(
                model=self.config.openai_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{data}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            raise ModelError(f"OpenAI request failed: {exc}") from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError(
                f"OpenAI response was not valid JSON: {text[:200]}"
            ) from exc

    def _client_or_create(self) -> Any:
        if self._client is None:
            api_key = self.config.api_key_for("openai")
            if not api_key:
                raise ModelError(
                    "OpenAI credentials required. Set OPENAI_API_KEY."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        return self._client


__all__ = ["OpenAIClient"]
