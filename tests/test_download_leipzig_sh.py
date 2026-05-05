"""Tests for scripts/download_leipzig.sh.

Tests cover:
- Script exists and is executable
- Bash syntax is valid (bash -n)
- .gitignore contains data/leipzig/
- All three corpus IDs are present in the script
- URL pattern references the correct host
- SHA-256 TODO placeholder pattern is present (first-run print behaviour)
- Idempotency guards are present (skip-if-exists logic for both tarball and
  extracted sentences file)
- Lang dir names are en/ru/es (NOT eng/rus/spa)
- Tarballs are kept in data/leipzig/_downloads/ for re-extract
- SCRIPT_DIR / REPO_ROOT self-discovery pattern is present
- SKIP_SHA env-var hook is present

Note: the script is NOT executed (doing so would download ~600 MB).
Idempotency is verified by inspecting the script source for the expected
guard logic.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "download_leipzig.sh"
GITIGNORE = REPO_ROOT / ".gitignore"

CORPUS_IDS = [
    "eng_news_2019_1M",
    "eng_news_2020_1M",
    "eng_news_2023_1M",
    "rus_news_2019_1M",
    "rus_news_2020_1M",
    "rus_news_2023_1M",
    "spa_news_2019_1M",
    "spa_news_2020_1M",
    "spa_news_2023_1M",
]

LANG_DIRS = ["en", "ru", "es"]


# ---------------------------------------------------------------------------
# Existence and permissions
# ---------------------------------------------------------------------------


def test_script_exists():
    assert SCRIPT.exists(), f"Script not found: {SCRIPT}"


def test_script_is_executable():
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, f"Script not executable by owner: {SCRIPT}"


# ---------------------------------------------------------------------------
# Bash syntax
# ---------------------------------------------------------------------------


def test_bash_syntax():
    """bash -n must exit 0 — pure syntax check, no execution."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# .gitignore
# ---------------------------------------------------------------------------


def test_gitignore_has_leipzig_entry():
    """data/leipzig/ must be excluded from git."""
    text = GITIGNORE.read_text()
    assert "data/leipzig/" in text, (
        ".gitignore is missing data/leipzig/ — large corpus files would end up in git"
    )


# ---------------------------------------------------------------------------
# Corpus IDs and URL pattern
# ---------------------------------------------------------------------------


def test_all_corpus_ids_present():
    """All three pinned corpus IDs must appear in the script."""
    text = SCRIPT.read_text()
    for corpus_id in CORPUS_IDS:
        assert corpus_id in text, (
            f"Corpus ID '{corpus_id}' not found in {SCRIPT}"
        )


def test_url_pattern_correct_host():
    """URL must use downloads.wortschatz-leipzig.de domain."""
    text = SCRIPT.read_text()
    assert "wortschatz-leipzig.de" in text, (
        "Expected URL host 'wortschatz-leipzig.de' not found in script"
    )


def test_url_pattern_uses_tar_gz():
    """Tarballs are .tar.gz format."""
    text = SCRIPT.read_text()
    assert ".tar.gz" in text, "Expected .tar.gz suffix in URL pattern"


# ---------------------------------------------------------------------------
# Lang dir names: en/ru/es not eng/rus/spa
# ---------------------------------------------------------------------------


def test_lang_dir_names_are_short():
    """data/leipzig/<lang>/ must use en/ru/es not eng/rus/spa."""
    text = SCRIPT.read_text()
    # Require the precise quoted form as it appears in the LANG_DIRS array.
    for lang in LANG_DIRS:
        assert f'"{lang}"' in text, (
            f'Short lang dir literal "{lang}" not found in LANG_DIRS array'
        )
    # Also assert the long forms are NOT used as dir names
    for bad in ["leipzig/eng", "leipzig/rus", "leipzig/spa"]:
        assert bad not in text, (
            f"Long lang dir form '{bad}' found — should use short form"
        )


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------


def test_downloads_subdir_referenced():
    """Tarballs should be kept in data/leipzig/_downloads/."""
    text = SCRIPT.read_text()
    assert "_downloads" in text, (
        "Expected _downloads subdir for tarball retention not found in script"
    )


def test_sentences_txt_extraction():
    """Script should extract -sentences.txt members from tarballs."""
    text = SCRIPT.read_text()
    assert "sentences.txt" in text, (
        "Expected extraction of -sentences.txt not found in script"
    )


# ---------------------------------------------------------------------------
# SHA-256 placeholder pattern
# ---------------------------------------------------------------------------


def test_sha256_verify_or_print_logic_present():
    """Script must have the verify_or_print helper that handles TODO placeholders.

    This test verifies the pattern — not the placeholder state — so it passes
    both during Phase A (when some entries are still TODO) and after Phase B
    (when all sha256s are pinned).  The verify_or_print function handles
    TODO-vs-real logic; asserting it exists is sufficient to confirm the
    first-run hash-printing behaviour is in place.
    """
    text = SCRIPT.read_text()
    assert "verify_or_print" in text, (
        "verify_or_print helper not found in script — "
        "first-run SHA-256 placeholder logic is missing"
    )

def test_all_corpus_ids_have_sha256_entry():
    """Every corpus ID in CORPUS_IDS must have a corresponding entry in EXPECTED_SHA256."""
    text = SCRIPT.read_text()
    for corpus_id in CORPUS_IDS:
        assert f'["{corpus_id}"]' in text, (
            f"EXPECTED_SHA256 entry missing for corpus ID '{corpus_id}'"
        )


def test_skip_sha_env_var_present():
    """SKIP_SHA env-var hook must be present to allow bypassing verification."""
    text = SCRIPT.read_text()
    assert "SKIP_SHA" in text, (
        "Expected SKIP_SHA env-var hook not found in script"
    )


# ---------------------------------------------------------------------------
# Idempotency guards
# ---------------------------------------------------------------------------


def test_idempotency_skip_if_tarball_exists():
    """Script must check for existing tarball before downloading."""
    text = SCRIPT.read_text()
    # Match the actual guard, not 'rm -f' inside trap cleanups.
    assert '[[ -f "$tarball_dest" ]]' in text, (
        "Tarball idempotency guard '[[ -f \"$tarball_dest\" ]]' not found"
    )


def test_idempotency_skip_if_sentences_file_exists():
    """Script must check for final sentences file before extracting."""
    text = SCRIPT.read_text()
    # Match the actual guard pattern. Both -f and -s are required since the
    # script also rejects an empty file as an invalid prior extract.
    assert '[[ -f "$sentences_file" && -s "$sentences_file" ]]' in text, (
        "Sentences-file idempotency guard "
        "'[[ -f \"$sentences_file\" && -s \"$sentences_file\" ]]' not found"
    )


# ---------------------------------------------------------------------------
# SCRIPT_DIR / REPO_ROOT self-discovery
# ---------------------------------------------------------------------------


def test_script_dir_discovery():
    """Script must use BASH_SOURCE-based self-discovery for SCRIPT_DIR."""
    text = SCRIPT.read_text()
    assert "BASH_SOURCE" in text, (
        "Expected BASH_SOURCE-based SCRIPT_DIR discovery not found"
    )
    assert "SCRIPT_DIR" in text, "SCRIPT_DIR variable not found in script"
    assert "REPO_ROOT" in text, "REPO_ROOT variable not found in script"


# ---------------------------------------------------------------------------
# set -euo pipefail
# ---------------------------------------------------------------------------


def test_strict_mode():
    """Script must use set -euo pipefail."""
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text, (
        "Expected 'set -euo pipefail' not found in script"
    )


# ---------------------------------------------------------------------------
# Dependency checks (wget/curl, tar, sha256sum)
# ---------------------------------------------------------------------------


def test_dependency_check_wget_or_curl():
    """Script must verify wget or curl availability."""
    text = SCRIPT.read_text()
    has_wget = "wget" in text
    has_curl = "curl" in text
    assert has_wget or has_curl, (
        "Script should use wget or curl for downloads"
    )


def test_dependency_check_tar():
    """Script must use tar for extraction."""
    text = SCRIPT.read_text()
    assert "tar" in text, "Expected tar usage not found in script"


# ---------------------------------------------------------------------------
# Atomic .tmp staging and cleanup trap
# ---------------------------------------------------------------------------


def test_tmp_staging_present():
    """Downloads must use .tmp staging files for atomic moves."""
    text = SCRIPT.read_text()
    assert ".tmp" in text, (
        "Expected .tmp staging suffix not found — atomic download pattern missing"
    )


def test_trap_cleanup_present():
    """trap must be used to clean up .tmp files on error."""
    text = SCRIPT.read_text()
    assert "trap" in text, (
        "Expected trap cleanup for .tmp files not found in script"
    )
