# docs/

Textbook-style overview of ph-project: a 9-chapter document explaining the cross-linguistic semantic comparison study from motivation through results.

## Building

```bash
cd docs/
make pdf
```

This runs `latexmk` (configured by `.latexmkrc` to use lualatex + biber).

Output: `overview.pdf`

### Prerequisites

- TeX Live (lualatex, latexmk, biber). On Debian/Ubuntu:
  `sudo apt install texlive-luatex texlive-bibtex-extra biber`
  Or with TinyTeX: `tlmgr install biber biblatex` (TinyTeX is minimal by default).
- Fonts: DejaVu Serif and DejaVu Sans Mono (Cyrillic fallback and box-drawing characters in the codebase tour). On Debian/Ubuntu: `sudo apt install fonts-dejavu`.

## Structure

- `overview.tex` — main document (title page, TOC, chapter inputs, bibliography)
- `01-motivation.tex` through `09-results.tex` — chapter files
- `refs.bib` — shared bibliography (add entries here as chapters are written)
- `figures/` — TikZ and image files for diagrams
- `STYLE.md` — voice, terminology, and formatting conventions (read before writing)
- `Makefile` — build targets (`pdf`, `clean`)

## Style

Read `STYLE.md` before writing or editing any chapter. It covers voice, banned constructions, citation conventions, math typesetting, and terminology consistency.
