"""
baselines/vectors.py
--------------------
Load fastText word vectors for the ph-project Phase 2 baselines.

Usage
-----
    from baselines.vectors import load_fasttext

    # Language-specific CC-300 (binary .bin), returns gensim KeyedVectors
    kv = load_fasttext("en", "cc")

    # MUSE supervised-aligned vectors (.vec text), returns gensim KeyedVectors
    kv = load_fasttext("ru", "aligned")

Environment
-----------
FASTTEXT_DATA_DIR — override the root directory for fastText files.
Default: <this file's package root> / "data/fasttext"
(i.e. ph-project/data/fasttext when installed from the repo root).

Data layout expected
--------------------
$FASTTEXT_DATA_DIR/
    cc/
        cc.en.300.bin
        cc.ru.300.bin
        cc.es.300.bin
    aligned/
        wiki.multi.en.vec
        wiki.multi.ru.vec
        wiki.multi.es.vec

Run scripts/download_fasttext.sh to populate these files.
"""
import os
import pathlib

from gensim.models import KeyedVectors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_LANGS = frozenset({"en", "ru", "es"})
_VALID_KINDS = frozenset({"cc", "aligned"})

# Default data root: resolve relative to the repo root (two levels up from
# this file's package directory).
_DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "fasttext"


def _data_dir() -> pathlib.Path:
    """Return the resolved fastText data root, honouring FASTTEXT_DATA_DIR."""
    env = os.environ.get("FASTTEXT_DATA_DIR")
    if env:
        return pathlib.Path(env)
    return _DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_fasttext(lang: str, kind: str) -> KeyedVectors:
    """Load fastText vectors for *lang* and *kind*, returning a KeyedVectors.

    Parameters
    ----------
    lang : str
        Language code.  Must be one of ``{"en", "ru", "es"}``.
    kind : str
        Embedding flavour.  Must be one of ``{"cc", "aligned"}``:

        * ``"cc"``      — fastText Common Crawl 300-dim ``.bin`` (binary).
          Loaded via ``gensim.models.fasttext.load_facebook_model(...).wv``.
          Supports subword OOV composition.
        * ``"aligned"`` — MUSE supervised-aligned wiki vectors ``.vec`` (text).
          Loaded via ``KeyedVectors.load_word2vec_format(..., binary=False)``.
          No subword fallback; OOV words will raise a KeyError at lookup.

    Returns
    -------
    gensim.models.KeyedVectors
        300-dimensional word vectors ready for ``kv[word]`` lookup.

    Raises
    ------
    ValueError
        If *lang* or *kind* is not in the allowed set.
    FileNotFoundError
        If the expected file is not present.  Run
        ``scripts/download_fasttext.sh`` to download the vectors.
    """
    if lang not in _VALID_LANGS:
        raise ValueError(
            f"lang must be one of {sorted(_VALID_LANGS)!r}, got {lang!r}."
        )
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_VALID_KINDS)!r}, got {kind!r}."
        )

    root = _data_dir()

    if kind == "cc":
        return _load_cc(lang, root)
    else:  # kind == "aligned"
        return _load_aligned(lang, root)


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_cc(lang: str, root: pathlib.Path) -> KeyedVectors:
    """Load a CC-300 .bin file via gensim's Facebook fastText loader."""
    filename = f"cc.{lang}.300.bin"
    path = root / "cc" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"fastText CC-300 file not found: {path}\n"
            f"Run scripts/download_fasttext.sh to download it."
        )

    # Import here so the module can be imported without gensim.models.fasttext
    # failing if gensim is partially installed (unlikely but defensive).
    from gensim.models.fasttext import load_facebook_model

    model = load_facebook_model(str(path))
    return model.wv


def _load_aligned(lang: str, root: pathlib.Path) -> KeyedVectors:
    """Load a MUSE wiki.multi .vec file via KeyedVectors.load_word2vec_format."""
    filename = f"wiki.multi.{lang}.vec"
    path = root / "aligned" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"MUSE aligned vectors file not found: {path}\n"
            f"Run scripts/download_fasttext.sh to download it."
        )

    kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
    return kv
