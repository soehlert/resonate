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
PROMOTABLE_GENRES: set[str] = set(_tax_data.get("promotable_genres", []))
PRIMARY_GENRE_STEMS: dict[str, list[str]] = _tax_data.get("primary_genre_stems", {})
FAMILY_TO_PRIMARY: dict[str, str] = _tax_data.get("family_to_primary", {})


def _parse_families(item: dict) -> tuple[str, ...]:
    fams = item.get("families")
    if fams:
        return tuple(fams) if isinstance(fams, list) else (str(fams),)
    fam = item.get("family")
    if fam:
        return tuple(fam) if isinstance(fam, list) else (str(fam),)
    return ()


@dataclass(frozen=True)
class SubgenreSpec:
    """Specification for a canonical subgenre in the taxonomy."""

    name: str
    families: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def family(self) -> str:
        """Primary genre family for this subgenre."""
        return self.families[0] if self.families else ""


SUBGENRE_REGISTRY: list[SubgenreSpec] = [
    SubgenreSpec(
        name=item["name"],
        families=_parse_families(item),
        aliases=tuple(item.get("aliases", [])),
        description=item.get("description", ""),
    )
    for item in _tax_data.get("subgenres", [])
]

DEFAULT_SUB_GENRES: list[str] = [spec.name for spec in SUBGENRE_REGISTRY]

SUB_GENRE_STEMS: dict[str, list[str]] = {
    spec.name: list(spec.aliases) for spec in SUBGENRE_REGISTRY if spec.aliases
}

SUBGENRE_TO_FAMILIES: dict[str, tuple[str, ...]] = {
    alias.lower(): spec.families
    for spec in SUBGENRE_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
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


def is_family_subgenre_alias(family: str | None, tag: str) -> bool:
    """Check if tag is an explicit canonical alias for a subgenre in the given family."""
    if not family:
        return False
    fam_lower = family.lower()
    tag_clean = tag.lower().strip()
    fams = SUBGENRE_TO_FAMILIES.get(tag_clean)
    if fams:
        return any(f.lower() == fam_lower for f in fams)
    return False


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
    "PROMOTABLE_GENRES",
    "SUB_GENRE_STEMS",
    "SUBGENRE_REGISTRY",
    "SUBGENRE_TO_FAMILIES",
    "SUBGENRE_TO_FAMILY",
    "SubgenreSpec",
    "is_family_subgenre_alias",
]
