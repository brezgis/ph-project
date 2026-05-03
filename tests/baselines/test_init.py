"""
Tests for baselines/__init__.py — SUPPORTED_LANGUAGES constant.

Also covers the HEAD_POSITION invariant in baselines/distances.py,
making the import-time assertion explicit and survivable under -O.
"""

# ---------------------------------------------------------------------------
# SUPPORTED_LANGUAGES
# ---------------------------------------------------------------------------

class TestSupportedLanguages:
    def test_supported_languages_exists(self):
        """SUPPORTED_LANGUAGES is exported from the baselines package."""
        from baselines import SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES is not None

    def test_supported_languages_is_frozenset(self):
        """SUPPORTED_LANGUAGES must be a frozenset (immutable)."""
        from baselines import SUPPORTED_LANGUAGES
        assert isinstance(SUPPORTED_LANGUAGES, frozenset)

    def test_supported_languages_value(self):
        """SUPPORTED_LANGUAGES must equal exactly {'en', 'ru', 'es'}."""
        from baselines import SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES == {"en", "ru", "es"}

    def test_supported_languages_is_importable_directly(self):
        """Can import SUPPORTED_LANGUAGES without going through __init__ indirection."""
        from baselines import SUPPORTED_LANGUAGES as SL
        assert "en" in SL
        assert "ru" in SL
        assert "es" in SL
        assert len(SL) == 3


# ---------------------------------------------------------------------------
# HEAD_POSITION invariant (distances.py)
# Explicit test so the contract survives python -O (which strips assert stmts)
# ---------------------------------------------------------------------------

class TestHeadPositionInvariant:
    def test_head_position_keys_match_supported_languages(self):
        """HEAD_POSITION.keys() must equal set(SUPPORTED_LANGUAGES).

        This makes the distances.py module-level assertion explicit in the test
        suite so it survives python -O (which disables bare assert statements).
        """
        from baselines import SUPPORTED_LANGUAGES
        from baselines.distances import HEAD_POSITION

        assert HEAD_POSITION.keys() == set(SUPPORTED_LANGUAGES), (
            f"HEAD_POSITION keys {sorted(HEAD_POSITION)!r} must equal "
            f"SUPPORTED_LANGUAGES {sorted(SUPPORTED_LANGUAGES)!r}"
        )
