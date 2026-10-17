"""Shared helpers for walking Textract block relationship graphs."""

from __future__ import annotations

from typing import Any


def child_ids(block: dict[str, Any], relationship_type: str = "CHILD") -> list[str]:
    """Return IDs from relationships of the requested Textract relationship type."""
    return [
        identifier
        for relationship in block.get("Relationships", [])
        if isinstance(relationship, dict)
        and relationship.get("Type") == relationship_type
        and isinstance(relationship.get("Ids"), list)
        for identifier in relationship["Ids"]
        if isinstance(identifier, str)
    ]


def descendant_text(
    identifier: str,
    blocks_by_id: dict[str, dict[str, Any]],
    visited: set[str] | None = None,
) -> str:
    """Collect a block's text descendants, including selected checkboxes."""
    seen = set() if visited is None else visited
    if identifier in seen:
        return ""
    seen.add(identifier)
    block = blocks_by_id.get(identifier)
    if block is None:
        return ""
    text = block.get("Text")
    parts = [text] if isinstance(text, str) and text else []
    if block.get("BlockType") == "SELECTION_ELEMENT":
        status = block.get("SelectionStatus")
        if isinstance(status, str):
            parts.append(f"[{status.lower()}]")
    for child_identifier in child_ids(block):
        child_text = descendant_text(child_identifier, blocks_by_id, seen)
        if child_text:
            parts.append(child_text)
    return " ".join(parts)
