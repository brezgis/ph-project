# Canon term list schema

Each file lives at `canon-terms/<lang>/<domain>.yaml` where `<lang>` is one of
`en`, `ru`, `es` and `<domain>` is one of `color`, `emotion`, `kinship`.

Every term must trace to a specific publication. Deviations from cited sources
must be named and justified.

## Schema

```yaml
domain: <color|emotion|kinship>
language: <en|ru|es>
description: >
  One-paragraph prose statement of what this inventory is and how it was
  selected. Names the primary anchor publication and the scope rule (e.g.,
  "Berlin & Kay 1969 basic color terms plus the Russian blue split from
  Davidoff 1999").
sources:
  - citation: "Author Year. Title. Journal/Publisher."
    role: anchor | augmentation | critique
    notes: "what this source contributes (e.g., 'adds goluboj as an 11th BCT')"
terms:
  - term: "<surface form>"             # the string that will be matched in corpora
    gloss: "<English gloss>"            # optional for en, required for ru/es
    source: "Author Year"               # which cited source licenses this term
    notes: "optional — any caveat, register, or morphological note"
deviations:
  - item: "<what was changed>"
    from: "<what the source said>"
    reason: "<why we diverged>"
```

## Conventions

* Terms are lowercased, unaccented only where the language's own orthography
  omits the accent. Keep Spanish accents (`corazón`), Russian Cyrillic
  (`тоска`), English plain ASCII.
* Multi-word terms are allowed. See `ph-project-ssa.3` for how the head word
  is extracted per language.
* Source citations must be specific enough to locate the claim — page numbers
  where practical, otherwise section/chapter.
* If a term appears across multiple sources, list the earliest / anchor source
  in `source` and mention the others in `notes`.

## Citation short-form convention

The `source:` field on each term uses a short form that must resolve to exactly
one entry in the `sources:` list of the same file. The convention is:

* Single author: `"Surname Year"` — e.g., `"Murdock 1949"`.
* Two authors: `"Surname1 & Surname2 Year"` — e.g., `"Berlin & Kay 1969"`.
* Three or more authors: `"Surname1 et al. Year"` — e.g., `"Lillo et al. 2018"`.
* Collective authors (e.g., dictionaries): use the full institutional name —
  e.g., `"Real Academia Española 2014"`. Do not use acronyms like "RAE 2014".

Each short form must unambiguously identify a single `citation` entry (by
first author's surname plus year). If a file cites two works by the same
first author in the same year, disambiguate with `a`/`b` suffixes on the year.
