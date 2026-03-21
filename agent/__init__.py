"""
Ollama / local-LLM integration for iVISPAR.

`OllamaAgent` is lazy-imported so `import agent.CognitiveModel` works without
`iVISPAR/Source/Experiment` on PYTHONPATH (CognitiveModel has no Ollama dependency).
"""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = ["OllamaAgent"]


def __getattr__(name: str) -> Any:
    if name == "OllamaAgent":
        from .ollama_agent import OllamaAgent

        return OllamaAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | {"OllamaAgent"})
