# data/fasttext/

fastText vector files used in ph-project Phase 2 baselines.

These files are **not committed** (see `.gitignore`).
Run `scripts/download_fasttext.sh` from the repo root to download them.

---

## File list and source URLs

### CC-300 (language-specific, binary `.bin`)

| File | Source URL | Size (approx) |
|------|-----------|---------------|
| `cc/cc.en.300.bin` | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz | ~4.2 GB |
| `cc/cc.ru.300.bin` | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz | ~4.2 GB |
| `cc/cc.es.300.bin` | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz | ~4.2 GB |

Loaded via `gensim.models.fasttext.load_facebook_model(...).wv`.
Supports subword OOV composition — any word string returns a vector.

Reference: Grave et al. (2018) "Learning Word Vectors for 157 Languages".
https://fasttext.cc/docs/en/crawl-vectors.html

### MUSE supervised-aligned (cross-lingual, text `.vec`)

| File | Source URL | Size (approx) |
|------|-----------|---------------|
| `aligned/wiki.multi.en.vec` | https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec | ~650 MB |
| `aligned/wiki.multi.ru.vec` | https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ru.vec | ~650 MB |
| `aligned/wiki.multi.es.vec` | https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec | ~650 MB |

Loaded via `KeyedVectors.load_word2vec_format(..., binary=False)`.
**No subword fallback** — OOV words raise `KeyError` at lookup.

Reference: Conneau et al. (2018) "Word Translation Without Parallel Data" (MUSE).
https://github.com/facebookresearch/MUSE#download

---

## Expected SHA-256 checksums

Populated on first download. Until then the download script prints the actual
hash and prompts you to fill in the table below.

| File | SHA-256 |
|------|---------|
| `cc.en.300.bin` | 14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca |
| `cc.ru.300.bin` | 208df9419e13196de5b63008880999ebcf8383d083762c9bb0c210f84f280279 |
| `cc.es.300.bin` | b8c800affac505d60c8a929cb90ff2ef616b4e8b5224d8f3a0a5e911a8a6546e |
| `wiki.multi.en.vec` | b9558d40469e9ed6cb3963cc85b28a2e3841811a7d6d3b9ce3c54bf7784caacd |
| `wiki.multi.ru.vec` | 0cd989c36691df2d5ea64e0a59b9629df79e2faa48269ff41c2a2210bc6626eb |
| `wiki.multi.es.vec` | 96f1c2dd9651b653ebcd1fd720a20bb4195f9c457d9df4757768fc03b758a326 |

**How to populate:** after running `scripts/download_fasttext.sh`, it prints the
computed hash next to each `*** SHA-256 PLACEHOLDER ***` banner. Copy those
values into the table above and commit the update to `data/fasttext/README.md`.

---

## Expected directory layout

```
data/fasttext/
├── README.md               ← this file (committed)
├── cc/
│   ├── cc.en.300.bin       ← gitignored
│   ├── cc.ru.300.bin       ← gitignored
│   └── cc.es.300.bin       ← gitignored
└── aligned/
    ├── wiki.multi.en.vec   ← gitignored
    ├── wiki.multi.ru.vec   ← gitignored
    └── wiki.multi.es.vec   ← gitignored
```
