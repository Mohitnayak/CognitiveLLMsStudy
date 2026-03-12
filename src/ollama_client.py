"""
Thin wrapper around the Ollama Python library for chat/generate and model listing.
Supports image inputs (file path or base64) for VLMs (e.g. iVISPAR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama


def _response_to_dict(resp: Any) -> dict[str, Any]:
    """Normalize Ollama response to dict for consistent access (e.g. response['message']['content'])."""
    if isinstance(resp, dict):
        return resp
    return {
        "message": {
            "role": getattr(getattr(resp, "message", None), "role", "assistant"),
            "content": getattr(getattr(resp, "message", None), "content", "") or "",
        },
        "done": getattr(resp, "done", True),
    }


def list_models() -> list[dict[str, Any]]:
    """List models available in the local Ollama server."""
    response = ollama.list()
    return getattr(response, "models", []) or []


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Send a chat request to Ollama.

    Args:
        model: Model name (e.g. 'llava', 'qwen2-vl', 'qwen3-vl').
        messages: List of message dicts with 'role' and 'content'.
                  For vision models, a message can include 'images': [path_or_base64, ...].
        stream: If True, return an iterator of chunks; if False, return full response.

    Returns:
        Response dict with 'message' (and optionally 'done', etc.).
        For stream=True, returns a generator of response chunks.
    """
    if stream:
        return ollama.chat(model=model, messages=messages, stream=True)  # type: ignore
    out = ollama.chat(model=model, messages=messages)
    return _response_to_dict(out)


def chat_with_image(
    model: str,
    content: str,
    image_path: str | Path | None = None,
    image_base64: str | None = None,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Convenience: one user turn with optional image (for VLMs).

    Args:
        model: Model name (e.g. 'llava', 'qwen2-vl').
        content: Text prompt.
        image_path: Path to image file.
        image_base64: Base64-encoded image data (alternative to image_path).
        stream: Whether to stream the response.

    Returns:
        Same as chat().
    """
    images: list[str] = []
    if image_path is not None:
        images.append(str(Path(image_path).resolve()))
    if image_base64 is not None:
        images.append(image_base64)
    msg: dict[str, Any] = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    return chat(model=model, messages=[msg], stream=stream)


def generate(
    model: str,
    prompt: str,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Generate completion for a single prompt (no message history)."""
    if stream:
        return ollama.generate(model=model, prompt=prompt, stream=True)  # type: ignore
    return ollama.generate(model=model, prompt=prompt)
