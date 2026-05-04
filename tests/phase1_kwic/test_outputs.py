"""
tests/phase1_kwic/test_outputs.py
----------------------------------
Validate on-disk KWIC CSV + sidecar pairs against the rules in
data/kwic/SCHEMA.md "Validation rules (asserted in tests)" section.

Structure
---------
_assert_schema_compliant(csv_path, sidecar_path, lang, domain)
    Module-level helper that runs all assertions (a–h). Both the
    synthetic and real-data tests call this single function — DRY.

TestSyntheticCompliant
    Uses the `schema_compliant` fixture from conftest.py.
    The compliant pair passes every assertion.

TestSyntheticViolating
    Uses the `schema_violating` fixture from conftest.py.
    Each case is expected to raise AssertionError on exactly the
    rule it violates (verified via pytest.raises + match string).

TestRealData
    Gated on the presence of data/kwic/en/color.csv.
    Parameterized over all 9 (lang, domain) pairs.
    Each test loads the real CSV + sidecar and calls
    _assert_schema_compliant.

Assertion rules (per SCHEMA.md "Validation rules" section)
-----------------------------------------------------------
a. CSV columns are exactly [term, labels, sentence, target_idx,
   corpus_source] in that order.
b. labels == term row-for-row.
c. No NaN, empty, or whitespace-only sentence values.
d. Every term value in the CSV appears in the canon-terms YAML.
e. target_idx is integer dtype, 0 <= target_idx < len(sentence.split())
   for every row.
f. corpus_source matches CORPUS_SOURCE_ID[lang] for every row.
g. Language-specific character checks:
   - ru: every sentence has at least one Cyrillic char.
   - es: at least one accented char anywhere in the file.
   - en: >95% of chars are ASCII-ish (soft check).
h. Per-term row counts in CSV match n_emitted in the sidecar.
"""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd
import pytest
import yaml

from phase1_kwic.extract import CORPUS_SOURCE_ID

# ---------------------------------------------------------------------------
# Repo root — two parents up from this file
# (tests/phase1_kwic/test_outputs.py → tests/phase1_kwic/ → tests/ → repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CANON_DIR = REPO_ROOT / "canon-terms"

# ---------------------------------------------------------------------------
# Helper: load canon surfaces directly from YAML
# (faster than load_canon when we only need the surface set)
# ---------------------------------------------------------------------------

def _canon_surfaces(lang: str, domain: str) -> frozenset[str]:
    """Return the set of term surface strings from canon-terms/<lang>/<domain>.yaml."""
    yaml_path = _CANON_DIR / lang / f"{domain}.yaml"
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return frozenset(entry["term"] for entry in data.get("terms", []))


# ---------------------------------------------------------------------------
# Core assertion helper
# ---------------------------------------------------------------------------

def _assert_schema_compliant(
    csv_path: pathlib.Path,
    sidecar_path: pathlib.Path,
    lang: str,
    domain: str,
) -> None:
    """Assert that csv_path + sidecar_path conform to SCHEMA.md rules (a–h).

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the KWIC CSV file.
    sidecar_path : pathlib.Path
        Path to the matching .report.json sidecar.
    lang : str
        Language code ("en", "ru", or "es").
    domain : str
        Domain name ("color", "emotion", or "kinship").

    Raises
    ------
    AssertionError
        If any rule from SCHEMA.md "Validation rules" is violated.
    """
    # --- Load data --------------------------------------------------------
    df = pd.read_csv(csv_path, encoding="utf-8")
    with sidecar_path.open("r", encoding="utf-8") as fh:
        sidecar = json.load(fh)

    # (a) Column set and order --------------------------------------------
    expected_cols = ["term", "labels", "sentence", "target_idx", "corpus_source"]
    assert list(df.columns) == expected_cols, (
        f"rule(a): expected columns {expected_cols}, got {list(df.columns)}"
    )

    # (b) labels == term row-for-row --------------------------------------
    assert (df["labels"] == df["term"]).all(), (
        "rule(b): labels column must equal term column row-for-row; "
        f"differing rows: {df[df['labels'] != df['term']][['term','labels']].to_dict('records')}"
    )

    # (c) No NaN / empty / whitespace-only sentence -----------------------
    assert df["sentence"].notna().all(), (
        "rule(c): sentence column contains NaN values"
    )
    assert (df["sentence"].str.strip() != "").all(), (
        "rule(c): sentence column contains empty or whitespace-only values"
    )

    # (d) All term values appear in canon YAML ----------------------------
    canon = _canon_surfaces(lang, domain)
    unknown = set(df["term"].unique()) - canon
    assert not unknown, (
        f"rule(d): term values not in canon-terms/{lang}/{domain}.yaml: {sorted(unknown)}"
    )

    # (e) target_idx is integer dtype, 0 <= target_idx < n_tokens ---------
    assert pd.api.types.is_integer_dtype(df["target_idx"]), (
        f"rule(e): target_idx must be integer dtype, got {df['target_idx'].dtype}"
    )
    for i, row in df.iterrows():
        n_tokens = len(str(row["sentence"]).split())
        assert 0 <= row["target_idx"] < n_tokens, (
            f"rule(e): row {i}: target_idx={row['target_idx']} out of bounds "
            f"for sentence with {n_tokens} tokens: {row['sentence']!r}"
        )

    # (f) corpus_source matches pinned ID for lang ------------------------
    expected_source = CORPUS_SOURCE_ID[lang]
    assert (df["corpus_source"] == expected_source).all(), (
        f"rule(f): corpus_source must be {expected_source!r} for lang={lang!r}; "
        f"found: {df['corpus_source'].unique().tolist()}"
    )

    # (g) Language-specific character checks ------------------------------
    if lang == "ru":
        # Every sentence must contain at least one Cyrillic character
        cyrillic_re = re.compile(r"[Ѐ-ӿ]")
        bad_rows = df[~df["sentence"].str.contains(cyrillic_re)]
        assert len(bad_rows) == 0, (
            f"rule(g/ru): {len(bad_rows)} sentence(s) have no Cyrillic chars"
        )
    elif lang == "es":
        # At least one accented char anywhere in the file (soft, file-level)
        all_text = " ".join(df["sentence"].tolist())
        accented_re = re.compile(r"[áéíóúñÁÉÍÓÚÑüÜ¿¡]")
        assert accented_re.search(all_text) is not None, (
            "rule(g/es): no accented Spanish chars found anywhere in the CSV"
        )
    elif lang == "en":
        # >95% of chars are ASCII-ish (letter, digit, whitespace, common punct)
        all_text = " ".join(df["sentence"].tolist())
        total = len(all_text)
        if total > 0:
            ascii_re = re.compile(r'[A-Za-z0-9 \t\n\r.,;:!?"\'()\-—–…]')
            ascii_count = sum(1 for ch in all_text if ascii_re.match(ch))
            ratio = ascii_count / total
            assert ratio > 0.95, (
                f"rule(g/en): only {ratio:.1%} of chars are ASCII-ish (threshold 95%)"
            )

    # (h) Per-term row counts match n_emitted in sidecar ------------------
    term_counts = df["term"].value_counts().to_dict()
    for term_info in sidecar.get("terms", []):
        t = term_info["term"]
        n_emitted = term_info["n_emitted"]
        actual = term_counts.get(t, 0)
        assert actual == n_emitted, (
            f"rule(h): term {t!r}: sidecar says n_emitted={n_emitted} "
            f"but CSV has {actual} rows"
        )


# ---------------------------------------------------------------------------
# Synthetic-fixture tests
# ---------------------------------------------------------------------------

class TestSyntheticCompliant:
    """The schema_compliant fixture should pass all assertions."""

    def test_compliant_pair_passes(self, schema_compliant):
        csv_path, sidecar_path, lang, domain = schema_compliant
        # This must not raise — if it does, the fixture or validator is broken
        _assert_schema_compliant(csv_path, sidecar_path, lang, domain)


class TestSyntheticViolating:
    """Each schema_violating case must raise AssertionError on the right rule.

    We use pytest.raises(AssertionError, match=...) to verify:
    1. The validator DOES raise (not zero assertions fired).
    2. The message identifies the CORRECT rule (not a different one).

    If a violation is not detected at all, pytest.raises will fail with
    "DID NOT RAISE". If a different rule fires first, the match= pattern
    will not match and the test will fail — proving exactly one rule fires.
    """

    # Bug ID → regex pattern that must appear in the AssertionError message.
    # Patterns are anchored to the rule tag inserted by _assert_schema_compliant.
    _BUG_TO_RULE_PATTERN: dict[str, str] = {
        "missing_column":            r"rule\(a\)",
        "nan_in_sentence":           r"rule\(c\)",
        "whitespace_only_sentence":  r"rule\(c\)",
        "target_idx_out_of_bounds":  r"rule\(e\)",
        "target_idx_negative":       r"rule\(e\)",
        "unknown_term":              r"rule\(d\)",
        "labels_mismatch":           r"rule\(b\)",
        "wrong_corpus_source":       r"rule\(f\)",
        "ru_no_cyrillic":            r"rule\(g/ru\)",
        "count_mismatch":            r"rule\(h\)",
        "es_no_accent":              r"rule\(g/es\)",
        "en_too_much_non_ascii":     r"rule\(g/en\)",
    }

    def test_violation_raises_correct_rule(self, schema_violating, request):
        """Each violating fixture raises AssertionError matching the right rule tag."""
        csv_path, sidecar_path, lang, domain = schema_violating

        # The fixture param id is exactly the bug_id set via pytest.param(id=...)
        # node.callspec.id is the full parametrize id, e.g. "missing_column"
        bug_id = request.node.callspec.id
        # Strip any outer fixture-name prefix that pytest might add
        for candidate in self._BUG_TO_RULE_PATTERN:
            if candidate in bug_id:
                bug_id = candidate
                break
        else:
            pytest.skip(
                f"Unknown bug_id {bug_id!r} — update _BUG_TO_RULE_PATTERN"
            )

        expected_pattern = self._BUG_TO_RULE_PATTERN[bug_id]

        with pytest.raises(AssertionError, match=expected_pattern):
            _assert_schema_compliant(csv_path, sidecar_path, lang, domain)


# ---------------------------------------------------------------------------
# Real-data integration tests
# ---------------------------------------------------------------------------

_REAL_DATA_PAIRS = [
    ("en", "color"),
    ("en", "emotion"),
    ("en", "kinship"),
    ("ru", "color"),
    ("ru", "emotion"),
    ("ru", "kinship"),
    ("es", "color"),
    ("es", "emotion"),
    ("es", "kinship"),
]


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "kwic" / "en" / "color.csv").exists(),
    reason="KWIC data not extracted — run scripts/extract_kwic.py first",
)
class TestRealData:
    """Validate all 9 real (lang, domain) CSV + sidecar pairs against SCHEMA.md."""

    @pytest.mark.parametrize("lang,domain", _REAL_DATA_PAIRS)
    def test_real_csv_schema_compliant(self, lang, domain):
        """Load the real CSV + sidecar and assert full schema compliance."""
        kwic_dir = REPO_ROOT / "data" / "kwic" / lang
        csv_path = kwic_dir / f"{domain}.csv"
        sidecar_path = kwic_dir / f"{domain}.report.json"

        assert csv_path.exists(), (
            f"Real CSV not found at {csv_path} — run scripts/extract_kwic.py first"
        )
        assert sidecar_path.exists(), (
            f"Real sidecar not found at {sidecar_path}"
        )

        _assert_schema_compliant(csv_path, sidecar_path, lang, domain)
