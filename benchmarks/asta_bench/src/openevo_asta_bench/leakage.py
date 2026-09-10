from __future__ import annotations

from typing import Any


PROTECTED_METADATA_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "reference",
    "reference_answer",
    "target",
    "rubric",
    "paper_id",
    "paper_ids",
    "paper_title",
    "paper_titles",
    "source_file",
    "sheet_name",
    "row_id",
    "sequence",
}


def leakage_basis_from_sample(sample: dict[str, Any]) -> dict[str, list[str]]:
    basis: dict[str, list[str]] = {}
    _collect(sample, basis=basis, path=())
    return {key: sorted(set(values)) for key, values in basis.items() if values}


def forbidden_literals_from_leakage_basis(basis: dict[str, list[str]], *, min_length: int = 4) -> list[str]:
    literals: list[str] = []
    for values in basis.values():
        for value in values:
            stripped = value.strip()
            if len(stripped) >= min_length:
                literals.append(stripped)
    return sorted(set(literals))


def _collect(value: Any, *, basis: dict[str, list[str]], path: tuple[str, ...]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = path + (key_text,)
            if key_text.lower() in PROTECTED_METADATA_KEYS:
                _append_leaf_values(child, basis.setdefault(key_text.lower(), []))
            else:
                _collect(child, basis=basis, path=child_path)
    elif isinstance(value, list):
        for child in value:
            _collect(child, basis=basis, path=path)


def _append_leaf_values(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(str(value))
    elif isinstance(value, list):
        for item in value:
            _append_leaf_values(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _append_leaf_values(item, out)
