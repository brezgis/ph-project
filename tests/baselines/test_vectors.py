"""
Tests for baselines/vectors.py — load_fasttext(lang, kind) -> KeyedVectors.

Synthetic tests run without any downloaded files by monkeypatching the gensim
loaders. Smoke tests skip gracefully when real files are absent — run them
after executing scripts/download_fasttext.sh.
"""
import os
import pathlib
import types

import numpy as np
import pytest
from gensim.models import KeyedVectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(words: list[str], dim: int = 300) -> KeyedVectors:
    """Build a tiny in-memory KeyedVectors fixture."""
    kv = KeyedVectors(vector_size=dim)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((len(words), dim)).astype(np.float32)
    kv.add_vectors(words, vectors)
    return kv


# ---------------------------------------------------------------------------
# Phase 1 tests — these must FAIL before baselines/vectors.py exists
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_kind_raises_value_error(self):
        from baselines.vectors import load_fasttext
        with pytest.raises(ValueError, match="kind"):
            load_fasttext("en", "invalid")

    def test_invalid_lang_raises_value_error(self):
        from baselines.vectors import load_fasttext
        with pytest.raises(ValueError, match="lang"):
            load_fasttext("de", "cc")

    def test_valid_langs_are_en_ru_es(self):
        """load_fasttext should accept exactly en, ru, es — no TypeError."""
        from baselines.vectors import load_fasttext
        # We just verify that valid lang/kind don't raise ValueError; the actual
        # file will be missing so we expect FileNotFoundError (not ValueError).
        for lang in ("en", "ru", "es"):
            with pytest.raises((FileNotFoundError, OSError)):
                load_fasttext(lang, "cc")

    def test_valid_kinds_are_cc_and_aligned(self):
        from baselines.vectors import load_fasttext
        for kind in ("cc", "aligned"):
            with pytest.raises((FileNotFoundError, OSError)):
                load_fasttext("en", kind)


class TestMissingFile:
    def test_missing_cc_raises_file_not_found(self, tmp_path, monkeypatch):
        """When the .bin file is absent, load_fasttext raises FileNotFoundError."""
        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext
        with pytest.raises(FileNotFoundError, match=r"cc\.en\.300\.bin"):
            load_fasttext("en", "cc")

    def test_missing_aligned_raises_file_not_found(self, tmp_path, monkeypatch):
        """When the .vec file is absent, load_fasttext raises FileNotFoundError."""
        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext
        with pytest.raises(FileNotFoundError, match=r"wiki\.multi\.en\.vec"):
            load_fasttext("en", "aligned")


class TestAlignedBranch:
    """Monkeypatch KeyedVectors.load_word2vec_format to return a fixture."""

    def test_aligned_dispatch(self, tmp_path, monkeypatch):
        fixture = _make_kv(["cat", "кот", "gato"])

        # Create a placeholder file so the existence check passes
        aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir()
        (aligned_dir / "wiki.multi.en.vec").write_text("placeholder")

        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(
            "gensim.models.KeyedVectors.load_word2vec_format",
            classmethod(lambda cls, *a, **kw: fixture),
        )

        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext

        result = load_fasttext("en", "aligned")
        assert isinstance(result, KeyedVectors)
        assert result.vector_size == 300

    def test_aligned_returns_finite_vector(self, tmp_path, monkeypatch):
        fixture = _make_kv(["cat"])

        aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir()
        (aligned_dir / "wiki.multi.en.vec").write_text("placeholder")

        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(
            "gensim.models.KeyedVectors.load_word2vec_format",
            classmethod(lambda cls, *a, **kw: fixture),
        )

        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext

        kv = load_fasttext("en", "aligned")
        vec = kv["cat"]
        assert vec.shape == (300,)
        assert np.all(np.isfinite(vec))
        assert np.any(vec != 0)


class TestCCBranch:
    """Monkeypatch gensim.models.fasttext.load_facebook_model to return a fixture."""

    def _make_fake_model(self, kv: KeyedVectors):
        """Wrap kv in a minimal object with a .wv attribute."""
        obj = types.SimpleNamespace(wv=kv)
        return obj

    def test_cc_dispatch(self, tmp_path, monkeypatch):
        fixture_kv = _make_kv(["cat", "кот", "gato"])
        fake_model = self._make_fake_model(fixture_kv)

        cc_dir = tmp_path / "cc"
        cc_dir.mkdir()
        (cc_dir / "cc.en.300.bin").write_bytes(b"placeholder")

        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(
            "gensim.models.fasttext.load_facebook_model",
            lambda *a, **kw: fake_model,
        )

        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext

        result = load_fasttext("en", "cc")
        assert isinstance(result, KeyedVectors)
        assert result.vector_size == 300

    def test_cc_returns_finite_vector(self, tmp_path, monkeypatch):
        fixture_kv = _make_kv(["cat"])
        fake_model = self._make_fake_model(fixture_kv)

        cc_dir = tmp_path / "cc"
        cc_dir.mkdir()
        (cc_dir / "cc.en.300.bin").write_bytes(b"placeholder")

        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(
            "gensim.models.fasttext.load_facebook_model",
            lambda *a, **kw: fake_model,
        )

        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext

        kv = load_fasttext("en", "cc")
        vec = kv["cat"]
        assert vec.shape == (300,)
        assert np.all(np.isfinite(vec))
        assert np.any(vec != 0)

    def test_cc_uses_load_facebook_model_not_word2vec(self, tmp_path, monkeypatch):
        """CC branch must call load_facebook_model, not load_word2vec_format."""
        fixture_kv = _make_kv(["cat"])
        fake_model = self._make_fake_model(fixture_kv)
        calls = {"cc_loader": 0, "w2v_loader": 0}

        cc_dir = tmp_path / "cc"
        cc_dir.mkdir()
        (cc_dir / "cc.en.300.bin").write_bytes(b"placeholder")

        def cc_loader(*a, **kw):
            calls["cc_loader"] += 1
            return fake_model

        monkeypatch.setenv("FASTTEXT_DATA_DIR", str(tmp_path))
        monkeypatch.setattr("gensim.models.fasttext.load_facebook_model", cc_loader)

        from baselines import vectors as v
        import importlib
        importlib.reload(v)
        from baselines.vectors import load_fasttext

        load_fasttext("en", "cc")
        assert calls["cc_loader"] == 1


# ---------------------------------------------------------------------------
# Smoke tests — skip if real files are absent
# ---------------------------------------------------------------------------

FASTTEXT_DATA_DIR = pathlib.Path(
    os.environ.get("FASTTEXT_DATA_DIR", "/home/anna/ph-project/data/fasttext")
)
_SKIP_MSG = (
    "Real fastText files not found. "
    "Run scripts/download_fasttext.sh first, then re-run this test."
)

SMOKE_PARAMS = [
    ("en", "cc",      "cc/cc.en.300.bin"),
    ("ru", "cc",      "cc/cc.ru.300.bin"),
    ("es", "cc",      "cc/cc.es.300.bin"),
    ("en", "aligned", "aligned/wiki.multi.en.vec"),
    ("ru", "aligned", "aligned/wiki.multi.ru.vec"),
    ("es", "aligned", "aligned/wiki.multi.es.vec"),
]

# Words that should be in-vocab for each language
SMOKE_WORDS = {
    "en": "cat",
    "ru": "кот",
    "es": "gato",
}


@pytest.mark.parametrize("lang,kind,rel_path", SMOKE_PARAMS)
def test_smoke_load(lang, kind, rel_path):
    file_path = FASTTEXT_DATA_DIR / rel_path
    if not file_path.exists():
        pytest.skip(f"{_SKIP_MSG} (missing: {file_path})")

    from baselines.vectors import load_fasttext
    kv = load_fasttext(lang, kind)
    assert isinstance(kv, KeyedVectors)
    assert kv.vector_size == 300

    word = SMOKE_WORDS[lang]
    assert word in kv, f"Expected '{word}' to be in vocab for {lang}/{kind}"
    vec = kv[word]
    assert vec.shape == (300,)
    assert np.all(np.isfinite(vec))
    assert np.any(vec != 0), f"Zero vector returned for '{word}' in {lang}/{kind}"
