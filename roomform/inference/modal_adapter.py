"""Shared Modal adapter for pipeline stages.

Every remote stage follows the same shape: an image, a GPU function
with bytes/paths in and a serializable result out, and a local
entrypoint that moves files. This module owns those conventions so a
stage's ``modal_app.py`` declares only what is unique to it — the
image layers and the function body.

Usage::

    from roomform.modal_adapter import StageApp, torch_image

    stage = StageApp("lifting-spatiallm")

    @stage.gpu(image=torch_image("transformers"), timeout=600)
    def lift(point_cloud_bytes: bytes) -> str: ...

    @stage.entrypoint()
    def main(point_cloud: str, out: str):
        stage.run_file(lift, point_cloud, out)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal

MAX_FILE_BYTES = 128 * 1024 * 1024
DEFAULT_GPU = "L4"
DEFAULT_TIMEOUT = 10 * 60

_BASE_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def torch_image(
    *pip: str,
    base: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
    apt: tuple[str, ...] = (
        "git",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
    ),
    env: dict[str, str] | None = None,
    run_commands: tuple[str, ...] = (),
    with_roomform: bool = False,
) -> modal.Image:
    """CUDA torch image with the stage's extra layers on top."""
    image = (
        modal.Image.from_registry(base)
        .env({**_BASE_ENV, **(env or {})})
        .apt_install(*apt)
    )
    if pip:
        image = image.pip_install(*pip)
    if run_commands:
        image = image.run_commands(*run_commands)
    if with_roomform:
        image = image.pip_install(
            "pydantic>=2", "trimesh"
        ).add_local_python_source("roomform")
    return image


def slim_image(*pip: str, with_roomform: bool = True) -> modal.Image:
    """Debian-slim image for stages that don't need the CUDA base."""
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy", "pydantic>=2", *pip
    )
    if with_roomform:
        image = image.add_local_python_source("roomform")
    return image


class StageApp:
    """One Modal app per pipeline stage, with shared conventions:
    ``roomform-<stage>`` naming, a per-stage volume at /vol, L4 default
    GPU, and file-transport helpers for entrypoints."""

    VOL = "/vol"

    def __init__(self, name: str):
        self.name = name
        self.app = modal.App(f"roomform-{name}")
        self.volume = modal.Volume.from_name(
            f"roomform-{name}", create_if_missing=True
        )

    def gpu(
        self,
        *,
        image: modal.Image,
        gpu: str = DEFAULT_GPU,
        timeout: int = DEFAULT_TIMEOUT,
        with_volume: bool = False,
        **kwargs: Any,
    ) -> Callable:
        """``app.function`` with stage defaults; ``with_volume`` mounts
        the stage volume at ``StageApp.VOL``."""
        if with_volume:
            kwargs.setdefault("volumes", {})[self.VOL] = self.volume
        return self.app.function(
            image=image, gpu=gpu, timeout=timeout, **kwargs
        )

    def cpu(self, *, image: modal.Image, **kwargs: Any) -> Callable:
        return self.gpu(image=image, gpu=None, **kwargs)  # type: ignore[arg-type]

    def entrypoint(self) -> Callable:
        return self.app.local_entrypoint()

    @staticmethod
    def read_input(path: str) -> bytes:
        data = Path(path).read_bytes()
        if not data:
            raise ValueError(f"{path} is empty")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(
                f"{path} exceeds the {MAX_FILE_BYTES // 2**20} MiB limit"
            )
        return data

    @staticmethod
    def write_output(out: str, result: str | bytes) -> Path:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(result, bytes):
            output.write_bytes(result)
        else:
            output.write_text(result, encoding="utf-8")
        return output

    def run_file(
        self, fn: Any, input_path: str, out: str, **kwargs: Any
    ) -> Path:
        """bytes-in/result-out convenience: read + guard the input,
        call ``fn.remote``, persist the result."""
        result = fn.remote(self.read_input(input_path), **kwargs)
        output = self.write_output(out, result)
        print(f"[roomform-{self.name}] wrote {output}")
        return output
