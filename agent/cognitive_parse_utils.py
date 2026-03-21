"""
Parse and validate CognitiveAgent move candidates.

Filters template/placeholder lines and enforces ICML-style commands so DQN
and Unity see well-formed ``move <color> <shape> <dir>`` strings.
"""

from __future__ import annotations

import re
from typing import List, Set

# Same 2D→3D mapping as LLMAgent.parse_action
SHAPE_MAP = {
    "circle": "sphere",
    "square": "cube",
    "triangle": "pyramid",
    "hexagon": "cylinder",
}

_PLACEHOLDER_FRAGMENTS = (
    "<object",
    "object color",
    "object shape",
    "<your",
    "placeholder",
    "example action",
)

_MOVE_RE = re.compile(
    r"^move\s+(green|red|blue|yellow)\s+(cube|sphere|pyramid|cylinder)\s+(up|down|left|right)\s*\.?\s*$",
    re.IGNORECASE,
)


def normalize_move_shapes(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for k, v in SHAPE_MAP.items():
        low = re.sub(rf"\b{re.escape(k)}\b", v, low)
    return low


def is_placeholder_junk(line: str) -> bool:
    low = line.lower()
    if "<" in line or ">" in line:
        return True
    return any(p in low for p in _PLACEHOLDER_FRAGMENTS)


def is_valid_move_command(line: str) -> bool:
    if not line or is_placeholder_junk(line):
        return False
    norm = normalize_move_shapes(line)
    if not norm.startswith("move "):
        return False
    return bool(_MOVE_RE.match(norm))


def first_line_only(s: str) -> str:
    return s.split("\n")[0].strip()


def strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^\s*[-*•]\s*", "", line.strip())


def _add_candidate(out: List[str], seen: Set[str], raw_fragment: str, max_candidates: int) -> bool:
    """Append one valid move if possible. Returns True iff len(out) >= max_candidates."""
    line = first_line_only(raw_fragment)
    line = strip_bullet_prefix(line)
    line = re.sub(r"(?i)^action\s*:\s*", "", line).strip()
    line = first_line_only(line)
    line = normalize_move_shapes(line)
    if not is_valid_move_command(line):
        return False
    key = line.lower()
    if key in seen:
        return False
    seen.add(key)
    out.append(line)
    return len(out) >= max_candidates


def extract_candidate_moves(response: str, max_candidates: int) -> List[str]:
    """
    Pull up to ``max_candidates`` distinct valid ``move ...`` lines.
    Accepts ``action: move ...``, bullet-prefixed lines, or plain ``move ...``.
    """
    if max_candidates <= 0:
        return []
    raw = response.replace("\r\n", "\n")
    out: List[str] = []
    seen: Set[str] = set()

    for m in re.finditer(r"(?is)action\s*:\s*(move\s+[^\n]+)", raw):
        if _add_candidate(out, seen, m.group(1).strip(), max_candidates):
            return out

    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if not re.match(r"(?i)^\s*[-*•]?\s*(action\s*:\s*)?move\s+", s):
            continue
        if _add_candidate(out, seen, s, max_candidates):
            return out

    return out


def sanitize_chosen_move(chosen: str) -> str:
    """Single line, no bullet, normalized shapes; does not guarantee validity."""
    s = first_line_only(chosen)
    s = strip_bullet_prefix(s)
    s = re.sub(r"(?i)^action\s*:\s*", "", s).strip()
    s = first_line_only(s)
    return normalize_move_shapes(s)
