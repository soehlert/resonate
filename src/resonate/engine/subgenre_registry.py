"""Canonical subgenre specifications, style families, and taxonomy registries."""

from __future__ import annotations

from dataclasses import dataclass, field

from resonate.config import load_data_file
from resonate.engine.tag_filter import (
    GENERIC_MODIFIERS,
    GENRE_STOP_WORDS,
    NATIONALITY_STRINGS,
)

_tax_data = load_data_file("taxonomy.yaml")

DEFAULT_PRIMARY_GENRES: list[str] = _tax_data.get("primary_genres", [])
PRIMARY_GENRE_STEMS: dict[str, list[str]] = _tax_data.get("primary_genre_stems", {})
FAMILY_TO_PRIMARY: dict[str, str] = _tax_data.get("family_to_primary", {})


@dataclass(frozen=True)
class SubgenreSpec:
    """Specification for a canonical subgenre in the taxonomy."""

    name: str
    family: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


SUBGENRE_REGISTRY: list[SubgenreSpec] = [
    SubgenreSpec(
        name=item["name"],
        family=item["family"],
        aliases=tuple(item.get("aliases", [])),
        description=item.get("description", ""),
    )
    for item in _tax_data.get("subgenres", [])
]

DEFAULT_SUB_GENRES: list[str] = [spec.name for spec in SUBGENRE_REGISTRY]

SUB_GENRE_STEMS: dict[str, list[str]] = {
    spec.name: list(spec.aliases) for spec in SUBGENRE_REGISTRY if spec.aliases
}

SUBGENRE_TO_FAMILY: dict[str, str] = {
    alias.lower(): spec.family
    for spec in SUBGENRE_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
}

COMPOUND_SUBGENRE_WHITELIST: set[str] = {
    alias.lower()
    for spec in SUBGENRE_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
    if " " in alias or "-" in alias
}

MUTUALLY_EXCLUSIVE_STYLES: list[set[str]] = [
    set(pair) for pair in _tax_data.get("mutually_exclusive_styles", [])
]

__all__ = [
    "COMPOUND_SUBGENRE_WHITELIST",
    "DEFAULT_PRIMARY_GENRES",
    "DEFAULT_SUB_GENRES",
    "FAMILY_TO_PRIMARY",
    "GENERIC_MODIFIERS",
    "GENRE_STOP_WORDS",
    "MUTUALLY_EXCLUSIVE_STYLES",
    "NATIONALITY_STRINGS",
    "PRIMARY_GENRE_STEMS",
    "SUB_GENRE_STEMS",
    "SUBGENRE_REGISTRY",
    "SUBGENRE_TO_FAMILY",
    "SubgenreSpec",
]
