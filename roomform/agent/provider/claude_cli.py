"""Claude Code CLI provider: headless `claude -p`, no API key needed.

Uses the machine's existing Claude subscription login. The card image
is passed by absolute path in the prompt; the model Reads it itself.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from roomform.agent.errors import ModelError


class ClaudeCLIClient:
    def judge(self, prompt: str, card_path: Path) -> dict[str, str]:
        """One single-shot headless call: card path + prompt -> verdict."""
        full = (
            f"Read the image file at {card_path.resolve()} and then "
            f"follow these instructions.\n\n{prompt}"
        )
        try:
            proc = subprocess.run(
                ["claude", "-p", full, "--output-format", "json"],
                capture_output=True,
                check=False,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            raise ModelError(
                "`claude` binary not found. Install Claude Code or use "
                "--provider anthropic|openai."
            ) from exc
        if proc.returncode != 0:
            raise ModelError(f"claude CLI failed: {proc.stderr.strip()[:500]}")
        try:
            envelope = json.loads(proc.stdout)
            result = str(envelope.get("result", ""))
        except json.JSONDecodeError:
            result = proc.stdout  # older CLIs may print bare text
        return _extract_verdict(result)


def _extract_verdict(text: str) -> dict[str, str]:
    """Pull {"verdict": ..., "reason": ...} out of possibly-noisy text."""
    for match in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    raise ModelError(f"claude CLI response had no verdict JSON: {text[:200]}")


__all__ = ["ClaudeCLIClient"]
