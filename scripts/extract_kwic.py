"""
scripts/extract_kwic.py
-----------------------
CLI wrapper for phase1_kwic.extract.extract_kwic.

Runs KWIC extraction for one (lang, domain) pair against one or more Leipzig
corpus files and writes:
  <output-dir>/<lang>/<domain>.csv
  <output-dir>/<lang>/<domain>.report.json

Usage
-----
    python -m scripts.extract_kwic --lang en --domain color
    python -m scripts.extract_kwic --lang ru --domain kinship --seed 42
    python -m scripts.extract_kwic --lang es --domain emotion --output-dir /tmp/kwic
    python -m scripts.extract_kwic --lang en --domain color \\
        --corpus-path data/leipzig/en/eng_news_2019_1M-sentences.txt \\
                      data/leipzig/en/eng_news_2020_1M-sentences.txt \\
                      data/leipzig/en/eng_news_2023_1M-sentences.txt \\
        --corpus-source-id eng_news_2019_1M eng_news_2020_1M eng_news_2023_1M

Options
-------
--lang        {en,ru,es}  Language code (required).
--domain      {color,emotion,kinship}  Semantic domain (required).
--corpus-path PATH [PATH ...]
                          One or more paths to Leipzig sentences TSV files.
                          Default: data/leipzig/<lang>/<corpus_id>-sentences.txt
                          for each ID in CORPUS_SOURCE_IDS[lang].
--corpus-source-id STR [STR ...]
                          One or more Leipzig corpus ID strings (must match
                          --corpus-path count when both are given).
                          Default: derived from the corpus path filename stems.
--n-samples INT           Max KWIC hits per term (default: 200).
--seed INT                Random seed for deterministic sampling (default: 0).
--window INT              ±window whitespace tokens for KWIC span (default: 10).
--min-post-target-tokens INT
                          Drop sentences with fewer post-target tokens (default: 5).
--output-dir DIR          Output directory root (default: data/kwic).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path when invoked as ``python -m scripts.extract_kwic``
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phase1_kwic.extract import extract_kwic, CORPUS_SOURCE_IDS, default_corpus_paths
from phase1_kwic import SUPPORTED_LANGUAGES, DOMAINS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DOMAINS),
        help="Semantic domain.",
    )
    parser.add_argument(
        "--corpus-path",
        type=pathlib.Path,
        nargs="+",
        default=None,
        help=(
            "One or more paths to Leipzig sentences TSV files. "
            "Default: data/leipzig/<lang>/<corpus_id>-sentences.txt "
            "for each ID in CORPUS_SOURCE_IDS[lang]."
        ),
    )
    parser.add_argument(
        "--corpus-source-id",
        nargs="+",
        default=None,
        help=(
            "One or more corpus ID strings for the 'corpus_source' column. "
            "Must match --corpus-path count when both are given. "
            "Default: derived from corpus path filename stems."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Maximum KWIC hits to emit per canon term (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic per-term subseeds (default: 0).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="KWIC window size in whitespace tokens on each side (default: 10).",
    )
    parser.add_argument(
        "--min-post-target-tokens",
        type=int,
        default=5,
        help=(
            "Drop sentences with fewer than this many whitespace tokens "
            "after the target (default: 5)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "data" / "kwic",
        help="Output directory root (default: data/kwic).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Resolve corpus paths
    if args.corpus_path is None:
        corpus_paths = default_corpus_paths(args.lang)
    else:
        corpus_paths = list(args.corpus_path)

    # Resolve corpus source IDs
    if args.corpus_source_id is not None:
        corpus_source_ids = list(args.corpus_source_id)
    else:
        # Derive IDs: use pinned list if it matches path count, else use stems
        pinned_ids = CORPUS_SOURCE_IDS.get(args.lang, [])
        if args.corpus_path is None:
            # Default paths → use the full pinned ID list
            corpus_source_ids = pinned_ids
        else:
            # Custom paths given, no IDs given → derive from filename stems
            corpus_source_ids = [p.stem for p in corpus_paths]

    # Validate equal lengths when both are explicitly supplied
    if args.corpus_path is not None and args.corpus_source_id is not None:
        if len(corpus_paths) != len(corpus_source_ids):
            print(
                f"ERROR: --corpus-path ({len(corpus_paths)} items) and "
                f"--corpus-source-id ({len(corpus_source_ids)} items) "
                f"must have the same number of arguments.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Check that all corpus files exist
    missing = [p for p in corpus_paths if not p.exists()]
    if missing:
        for p in missing:
            print(
                f"ERROR: Corpus file not found: {p}\n"
                f"Run scripts/download_leipzig.sh to download corpora first.",
                file=sys.stderr,
            )
        sys.exit(1)

    print(
        f"Extracting KWIC: lang={args.lang!r} domain={args.domain!r} "
        f"corpora={[str(p) for p in corpus_paths]} "
        f"n_samples={args.n_samples} seed={args.seed}"
    )

    df, report = extract_kwic(
        lang=args.lang,
        domain=args.domain,
        corpus_paths=corpus_paths,
        corpus_source_ids=corpus_source_ids,
        n_samples=args.n_samples,
        seed=args.seed,
        window=args.window,
        min_post_target_tokens=args.min_post_target_tokens,
    )

    # Write outputs
    out_dir = args.output_dir / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.domain}.csv"
    report_path = out_dir / f"{args.domain}.report.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # Summary
    total_rows = len(df)
    n_under = sum(1 for t in report["terms"] if t["under_target"])
    n_terms = len(report["terms"])
    print(
        f"  Wrote {total_rows} rows across {n_terms} terms "
        f"({n_under} under-target) → {csv_path}"
    )
    print(f"  Report → {report_path}")

    if n_under:
        print(f"  Under-target terms:")
        for t in report["terms"]:
            if t["under_target"]:
                print(
                    f"    {t['term']!r}: "
                    f"{t['n_emitted']} emitted / {t['n_corpus_hits']} hits"
                )


if __name__ == "__main__":
    main()
