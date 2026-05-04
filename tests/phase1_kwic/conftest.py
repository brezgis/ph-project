"""
Synthetic fixture builder for tests/phase1_kwic/test_outputs.py.

Two parameterized fixtures:
  - schema_compliant:   minimal valid CSV + sidecar pair; returns (csv_path, sidecar_path).
  - schema_violating:   one intentional bug per parameter case.

Design principles
-----------------
- The compliant baseline uses lang="en", domain="color" with real terms
  ("red", "blue") and the real corpus_source ID so non-targeted assertions
  stay green when we mutate a single field in each violation case.
- Each schema_violating case must fail EXACTLY ONE assertion in
  _assert_schema_compliant (not zero, not two).
- The "ru_no_cyrillic" violation switches lang to "ru" so the corpus_source
  and term assertions still pass, but the Cyrillic check fires.
- CSV and sidecar are written to pytest's tmp_path, so there is no disk
  state leakage between tests.
"""
from __future__ import annotations

import json
import pathlib
import textwrap

import pandas as pd
import pytest

from phase1_kwic.extract import CORPUS_SOURCE_IDS

# ---------------------------------------------------------------------------
# Constants shared by both fixtures
# ---------------------------------------------------------------------------

# Compliant baseline language / domain
_LANG = "en"
_DOMAIN = "color"
# Pick the first ID from the multi-year list as the single corpus ID for
# synthetic fixtures — all fixture rows use one deterministic ID so that
# the "wrong corpus source" violation test can supply a different lang's ID.
_CORPUS_SOURCE = CORPUS_SOURCE_IDS[_LANG][0]

# A pair of real English color terms (surfaces exactly as in the YAML)
_TERM_A = "red"
_TERM_B = "blue"

# Minimal compliant rows — target_idx is valid for the sentence token list
# Sentence has 7 tokens; "red" is at index 3, "blue" is at index 3.
_COMPLIANT_ROWS = [
    # (term, labels, sentence, target_idx, corpus_source)
    (_TERM_A, _TERM_A, "the quick brown red fox trotted away", 3, _CORPUS_SOURCE),
    (_TERM_A, _TERM_A, "a big bright red balloon floated up", 3, _CORPUS_SOURCE),
    (_TERM_A, _TERM_A, "she wore a red dress to school", 3, _CORPUS_SOURCE),
    (_TERM_B, _TERM_B, "the blue sky stretched endlessly above", 1, _CORPUS_SOURCE),
    (_TERM_B, _TERM_B, "a deep blue ocean surrounded the island", 2, _CORPUS_SOURCE),
]

# Sidecar that matches the compliant rows exactly
_COMPLIANT_SIDECAR: dict = {
    "language": _LANG,
    "domain": _DOMAIN,
    "corpus_source": _CORPUS_SOURCE,
    "corpus_total_sentences": 1000000,
    "extracted_at": "2026-05-04T01:00:00Z",
    "seed": 0,
    "n_samples_target": 200,
    "window": {"left": 10, "right": 10, "unit": "whitespace_tokens"},
    "min_post_target_tokens": 5,
    "matchers": {"en": "spacy:en_core_web_md", "ru": "pymorphy3==2.0.6"},
    "terms": [
        {
            "term": _TERM_A,
            "n_corpus_hits": 500,
            "n_kept_after_dedup": 498,
            "n_emitted": 3,       # matches 3 "red" rows above
            "under_target": True,
        },
        {
            "term": _TERM_B,
            "n_corpus_hits": 300,
            "n_kept_after_dedup": 298,
            "n_emitted": 2,       # matches 2 "blue" rows above
            "under_target": True,
        },
    ],
}


def _write_csv(path: pathlib.Path, rows: list[tuple]) -> None:
    """Write rows to a CSV with the canonical column header."""
    df = pd.DataFrame(
        rows,
        columns=["term", "labels", "sentence", "target_idx", "corpus_source"],
    )
    df.to_csv(path, index=False, encoding="utf-8")


def _write_sidecar(path: pathlib.Path, sidecar: dict) -> None:
    """Write sidecar dict as JSON."""
    path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# schema_compliant fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def schema_compliant(tmp_path: pathlib.Path):
    """Write a minimal valid CSV + sidecar to tmp_path.

    Returns (csv_path, sidecar_path, lang, domain).
    """
    csv_path = tmp_path / "color.csv"
    sidecar_path = tmp_path / "color.report.json"
    _write_csv(csv_path, _COMPLIANT_ROWS)
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


# ---------------------------------------------------------------------------
# Violation parameter definitions
# ---------------------------------------------------------------------------
#
# Each entry is (bug_id, description, mutate_fn) where:
#   bug_id       — short string used to identify the violated rule
#   mutate_fn(rows, sidecar, tmp_path) -> (csv_path, sidecar_path, lang, domain)
#
# Convention: keep all fields correct EXCEPT the one being tested.

def _violation_missing_column(tmp_path: pathlib.Path):
    """Drop the 'labels' column — violates rule (a): wrong column set."""
    df = pd.DataFrame(
        _COMPLIANT_ROWS,
        columns=["term", "labels", "sentence", "target_idx", "corpus_source"],
    )
    df = df.drop(columns=["labels"])
    csv_path = tmp_path / "color.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_nan_in_sentence(tmp_path: pathlib.Path):
    """Write a CSV where one sentence cell is literally empty string (roundtrips to NaN).

    Rule (c): NaN / empty sentence.
    """
    rows = list(_COMPLIANT_ROWS)
    # Replace the sentence in row 1 with empty string — pandas reads "" as NaN
    r = rows[1]
    rows[1] = (r[0], r[1], "", r[3], r[4])
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_whitespace_only_sentence(tmp_path: pathlib.Path):
    """Replace one sentence with whitespace-only string — violates rule (c)."""
    rows = list(_COMPLIANT_ROWS)
    r = rows[0]
    rows[0] = (r[0], r[1], "   \t  ", r[3], r[4])
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_target_idx_out_of_bounds(tmp_path: pathlib.Path):
    """Set target_idx to len(sentence.split()) — one past the end — violates rule (e)."""
    rows = list(_COMPLIANT_ROWS)
    r = rows[0]
    out_of_bounds_idx = len(r[2].split())  # == 7 for the 7-token sentence
    rows[0] = (r[0], r[1], r[2], out_of_bounds_idx, r[4])
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_target_idx_negative(tmp_path: pathlib.Path):
    """Set target_idx to -1 — violates rule (e)."""
    rows = list(_COMPLIANT_ROWS)
    r = rows[0]
    rows[0] = (r[0], r[1], r[2], -1, r[4])
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_unknown_term(tmp_path: pathlib.Path):
    """Use a term not in the canon YAML — violates rule (d)."""
    rows = list(_COMPLIANT_ROWS)
    # Replace "red" rows with an unknown term "fuchsia" (not in en/color.yaml)
    rows = [
        ("fuchsia", "fuchsia", r[2], r[3], r[4]) if r[0] == _TERM_A else r
        for r in rows
    ]
    # Update sidecar to match (term counts still consistent to isolate rule d)
    sidecar = json.loads(json.dumps(_COMPLIANT_SIDECAR))
    for entry in sidecar["terms"]:
        if entry["term"] == _TERM_A:
            entry["term"] = "fuchsia"
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_labels_mismatch(tmp_path: pathlib.Path):
    """labels != term on one row — violates rule (b)."""
    rows = list(_COMPLIANT_ROWS)
    r = rows[0]
    rows[0] = (r[0], "WRONG_LABEL", r[2], r[3], r[4])
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, _COMPLIANT_SIDECAR)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_wrong_corpus_source(tmp_path: pathlib.Path):
    """corpus_source is the Russian ID instead of English — violates rule (f)."""
    rows = [
        (r[0], r[1], r[2], r[3], CORPUS_SOURCE_IDS["ru"][0])  # wrong lang corpus
        for r in _COMPLIANT_ROWS
    ]
    sidecar = json.loads(json.dumps(_COMPLIANT_SIDECAR))
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_ru_no_cyrillic(tmp_path: pathlib.Path):
    """Russian CSV with no Cyrillic chars in any sentence — violates rule (g)."""
    # Use real ru corpus_source and a real ru color term "красный" surface form
    # but write only ASCII sentences so no Cyrillic chars appear.
    ru_corpus_source = CORPUS_SOURCE_IDS["ru"][0]
    # "красный" is the first term in ru/color.yaml
    ru_term = "красный"
    rows = [
        (ru_term, ru_term, "the quick brown fox jumped over", 3, ru_corpus_source),
        (ru_term, ru_term, "a bright sunny day in the park", 4, ru_corpus_source),
        (ru_term, ru_term, "she went to the store quickly", 1, ru_corpus_source),
    ]
    sidecar = {
        "language": "ru",
        "domain": "color",
        "corpus_source": ru_corpus_source,
        "corpus_total_sentences": 1000000,
        "extracted_at": "2026-05-04T01:00:00Z",
        "seed": 0,
        "n_samples_target": 200,
        "window": {"left": 10, "right": 10, "unit": "whitespace_tokens"},
        "min_post_target_tokens": 5,
        "matchers": {"ru": "pymorphy3==2.0.6"},
        "terms": [
            {
                "term": ru_term,
                "n_corpus_hits": 100,
                "n_kept_after_dedup": 100,
                "n_emitted": 3,
                "under_target": True,
            }
        ],
    }
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, "ru", "color"


def _violation_count_mismatch(tmp_path: pathlib.Path):
    """n_emitted in sidecar says 99 for 'red' but CSV only has 3 rows — violates rule (h)."""
    sidecar = json.loads(json.dumps(_COMPLIANT_SIDECAR))
    for entry in sidecar["terms"]:
        if entry["term"] == _TERM_A:
            entry["n_emitted"] = 99  # lies: says 99 but CSV has 3 rows
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, _COMPLIANT_ROWS)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, _LANG, _DOMAIN


def _violation_es_no_accent(tmp_path: pathlib.Path):
    """Spanish CSV with zero accented characters anywhere — violates rule (g/es).

    The validator's es branch requires at least one char from
    [áéíóúñÁÉÍÓÚÑüÜ¿¡] across the entire concatenated text.
    We use real es color terms (rojo, azul, negro) and the real es
    corpus_source but write sentences in plain ASCII Spanish so no
    accented char appears.
    """
    import json as _json

    es_corpus_source = CORPUS_SOURCE_IDS["es"][0]
    es_term_a = "rojo"
    es_term_b = "azul"
    rows = [
        (es_term_a, es_term_a, "el gato es rojo hoy", 3, es_corpus_source),
        (es_term_a, es_term_a, "un rojo brillante en la sala", 1, es_corpus_source),
        (es_term_b, es_term_b, "el azul del cielo es claro", 2, es_corpus_source),
    ]
    sidecar = {
        "language": "es",
        "domain": "color",
        "corpus_source": es_corpus_source,
        "corpus_total_sentences": 1000000,
        "extracted_at": "2026-05-04T01:00:00Z",
        "seed": 0,
        "n_samples_target": 200,
        "window": {"left": 10, "right": 10, "unit": "whitespace_tokens"},
        "min_post_target_tokens": 5,
        "matchers": {"es": "spacy:es_core_news_md"},
        "terms": [
            {
                "term": es_term_a,
                "n_corpus_hits": 200,
                "n_kept_after_dedup": 200,
                "n_emitted": 2,
                "under_target": True,
            },
            {
                "term": es_term_b,
                "n_corpus_hits": 150,
                "n_kept_after_dedup": 150,
                "n_emitted": 1,
                "under_target": True,
            },
        ],
    }
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, "es", "color"


def _violation_en_too_much_non_ascii(tmp_path: pathlib.Path):
    """English CSV where >5% of all chars are non-ASCII — violates rule (g/en).

    The validator's en branch checks that >95% of chars are ASCII-ish.
    We use real en color terms (red, blue) and the real en corpus_source,
    but pad each sentence with a long run of Cyrillic text so that the
    non-ASCII fraction well exceeds 5%.
    """
    # Build a long non-ASCII filler that will push the ratio below 95%
    filler = "ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ"  # 54 Cyrillic
    rows = [
        (_TERM_A, _TERM_A, f"the red car stopped {filler}", 1, _CORPUS_SOURCE),
        (_TERM_A, _TERM_A, f"a red flag waved {filler}", 1, _CORPUS_SOURCE),
        (_TERM_B, _TERM_B, f"the blue sky {filler}", 1, _CORPUS_SOURCE),
    ]
    sidecar = json.loads(json.dumps(_COMPLIANT_SIDECAR))
    # Fix term counts to match rows above
    for entry in sidecar["terms"]:
        if entry["term"] == _TERM_A:
            entry["n_emitted"] = 2
        elif entry["term"] == _TERM_B:
            entry["n_emitted"] = 1
    # Update: original sidecar has n_emitted=3 for red and 2 for blue;
    # we only emit 2 red and 1 blue here so we must also add a third row or fix
    # Add a third TERM_A row to keep n_emitted matching the sidecar's value
    rows.append((_TERM_A, _TERM_A, f"red means stop {filler}", 0, _CORPUS_SOURCE))
    for entry in sidecar["terms"]:
        if entry["term"] == _TERM_A:
            entry["n_emitted"] = 3
        elif entry["term"] == _TERM_B:
            entry["n_emitted"] = 1
    # Also fix blue count: sidecar says 2 blue, we have 1 blue row -> add one
    rows.append((_TERM_B, _TERM_B, f"blue water flowed {filler}", 0, _CORPUS_SOURCE))
    for entry in sidecar["terms"]:
        if entry["term"] == _TERM_B:
            entry["n_emitted"] = 2
    csv_path = tmp_path / "color.csv"
    _write_csv(csv_path, rows)
    sidecar_path = tmp_path / "color.report.json"
    _write_sidecar(sidecar_path, sidecar)
    return csv_path, sidecar_path, _LANG, _DOMAIN


# ---------------------------------------------------------------------------
# schema_violating fixture
# ---------------------------------------------------------------------------

_VIOLATIONS = [
    pytest.param(_violation_missing_column,           id="missing_column"),
    pytest.param(_violation_nan_in_sentence,          id="nan_in_sentence"),
    pytest.param(_violation_whitespace_only_sentence, id="whitespace_only_sentence"),
    pytest.param(_violation_target_idx_out_of_bounds, id="target_idx_out_of_bounds"),
    pytest.param(_violation_target_idx_negative,      id="target_idx_negative"),
    pytest.param(_violation_unknown_term,             id="unknown_term"),
    pytest.param(_violation_labels_mismatch,          id="labels_mismatch"),
    pytest.param(_violation_wrong_corpus_source,      id="wrong_corpus_source"),
    pytest.param(_violation_ru_no_cyrillic,           id="ru_no_cyrillic"),
    pytest.param(_violation_count_mismatch,           id="count_mismatch"),
    pytest.param(_violation_es_no_accent,             id="es_no_accent"),
    pytest.param(_violation_en_too_much_non_ascii,    id="en_too_much_non_ascii"),
]


@pytest.fixture(params=_VIOLATIONS)
def schema_violating(request, tmp_path: pathlib.Path):
    """Parameterized fixture: one intentional schema bug per case.

    Each case writes a CSV + sidecar with exactly ONE violation.
    The fixture returns (csv_path, sidecar_path, lang, domain).

    The test body (in test_outputs.py) is responsible for calling
    pytest.raises(AssertionError, match=...) to prove that exactly the
    right assertion fires.
    """
    mutate_fn = request.param
    return mutate_fn(tmp_path)
