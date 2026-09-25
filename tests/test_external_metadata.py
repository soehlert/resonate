"""Unit tests for external metadata sanitizers, aliases, and title matchers."""

from resonate.modules.external_metadata import (
    artist_matches,
    clean_retailer_noise,
    get_artist_aliases,
    uncensor_title,
)


def test_uncensor_title() -> None:
    """Test uncensoring masked profanities in track titles."""
    assert uncensor_title("Hatef--k") == "Hatefuck"
    assert uncensor_title("F**k You") == "Fuck You"
    assert uncensor_title("F*ck That") == "Fuck That"
    assert uncensor_title("Don't Sh*t Where You Eat") == "Don't Shit Where You Eat"
    assert uncensor_title("B*tch Please") == "Bitch Please"
    assert uncensor_title("Total A**hole") == "Total Asshole"
    assert uncensor_title("Clean Normal Title") == "Clean Normal Title"


def test_clean_retailer_noise() -> None:
    """Test stripping retailer promotional noise while preserving musical editions."""
    assert clean_retailer_noise("Stir The Blood (Best Buy Exclusive)") == "Stir The Blood"
    assert clean_retailer_noise("In Rainbows [Target Exclusive]") == "In Rainbows"
    assert clean_retailer_noise("Album (Walmart Edition)") == "Album"
    assert clean_retailer_noise("Album (iTunes Exclusive)") == "Album"
    assert clean_retailer_noise("Album (Amazon Exclusive)") == "Album"

    # Musical editions must be preserved!
    assert (
        clean_retailer_noise("Stir The Blood (Deluxe Edition)") == "Stir The Blood (Deluxe Edition)"
    )
    assert clean_retailer_noise("Abbey Road (2019 Remaster)") == "Abbey Road (2019 Remaster)"
    assert clean_retailer_noise("Nevermind (Expanded Edition)") == "Nevermind (Expanded Edition)"


def test_artist_matches_compound_band_names() -> None:
    """Test artist_matches with articles, compound names, and mismatches."""
    # Direct match & case insensitivity
    assert artist_matches("Radiohead", "radiohead") is True

    # Leading article 'The'
    assert artist_matches("The Beatles", "Beatles") is True
    assert artist_matches("Beatles", "The Beatles") is True

    # Compound band name 'Jay & Americans' vs 'Jay & The Americans'
    assert artist_matches("Jay & Americans", "Jay & The Americans") is True
    assert artist_matches("Jay & The Americans", "Jay & Americans") is True
    assert artist_matches("Huey Lewis & News", "Huey Lewis & The News") is True
    assert artist_matches("Echo & Bunnymen", "Echo & The Bunnymen") is True

    # Anti-mismatch protection (e.g. Ye must NOT match Yes)
    assert artist_matches("Ye", "Yes") is False
    assert artist_matches("Yes", "Ye") is False
    assert artist_matches("The Who", "The Weeknd") is False

    # Alphanumeric spacing and punctuation variants
    assert artist_matches("Link Wray & His Raymen", "Link Wray & His Ray Men") is True
    assert artist_matches("Link Wray & His Ray Men", "Link Wray & His Raymen") is True
    assert artist_matches("ZZ Top", "Z.Z. Top") is True
    assert artist_matches("Z.Z. Top", "ZZ Top") is True


def test_get_artist_aliases_variants() -> None:
    """Test get_artist_aliases generates dictionary and grammatical variants."""

    # Dictionary alias from ARTIST_ALIASES
    ye_aliases = get_artist_aliases("Ye")
    assert "kanye west" in ye_aliases or "Kanye West" in ye_aliases
    assert "Ye" in ye_aliases

    # Compound band name
    jay_aliases = get_artist_aliases("Jay & Americans")
    assert "Jay & The Americans" in jay_aliases

    # Leading The (strip-only)
    cure_aliases = get_artist_aliases("The Cure")
    assert "Cure" in cure_aliases
    assert "The Cure" in cure_aliases

    # Non-The bands should not have 'The ' prepended
    whitesnake_aliases = get_artist_aliases("Whitesnake")
    assert "The Whitesnake" not in whitesnake_aliases

