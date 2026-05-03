# Consistency review -- textbook overview

Reviewer: Bea, 2026-05-03

Overall: this is remarkably coherent for nine chapters written by nine different agents. The voice holds, the math is consistent, the citations all resolve, and the narrative builds cleanly from motivation through methodology to honest results. The issues below are genuine but none threatens the document's core quality.

---

## BLOCKER

### B1. Ch 6, Section 6.2.2 (line 32): symmetrization direction is wrong

**Location:** `06-pipeline.tex:32`

**Issue:** The text says "The attention matrix is first made symmetric by taking the element-wise minimum of $W$ and $W^\top$ --- an edge appears when *either* direction exceeds the threshold, which is the most inclusive choice." This conflates the distance-frame and weight-frame operations. Taking the minimum of *weights* means an edge only appears when *both* directions are strong -- the *least* inclusive choice. Chapter 4 (Section 4.6, line 176) correctly handles this: it takes the minimum of *distances* ($D_{ij} = 1 - W'_{ij}$), which corresponds to the maximum of *weights*. The parenthetical claim "which is the most inclusive choice" is backwards.

**Suggested fix:** Change to: "The attention matrix is first converted to a distance matrix ($1 - W_{ij}$), then made symmetric by taking the element-wise minimum of $D$ and $D^\top$ --- so an edge appears in the Rips complex at the threshold where the stronger of the two directed attention scores is high enough."

---

## MAJOR

### M1. Ch 6, Section 6.7 vs Ch 9, Section 9.3.3: KWIC extraction status contradicts

**Location:** `06-pipeline.tex:103` vs `09-results.tex:61-63`

**Issue:** Chapter 6 says "Phase 1 --- KWIC extraction (epic 5f9): complete. Canon term lists for all nine (language, domain) combinations have been finalized and concordance sentences extracted from the Leipzig Corpora." Chapter 9 says "KWIC sentence extraction from Leipzig Corpora is planned but not yet executed as a batch process." These directly contradict each other on a factual question about project status.

**Suggested fix:** Determine which is accurate and align both chapters. If canon term lists are complete but KWIC sentence extraction has not been run, Ch 6 should say so. If extraction is complete, Ch 9 Section 9.3.3 should be removed or moved to "Completed work."

### M2. Ch 9, Section 9.2.2: `Table~\ref{tab:baseline-a}` uses `\ref` instead of `\autoref`

**Location:** `09-results.tex:16`

**Issue:** STYLE.md requires `\autoref` for all internal references. This is the only instance of bare `\ref` in the document. It renders as "Table 9.1" rather than the hyperlinked format that `\autoref` would produce.

**Suggested fix:** Change `Table~\ref{tab:baseline-a}` to `\autoref{tab:baseline-a}`.

### M3. Ch 7: length exceeds STYLE.md target

**Location:** `07-codebase.tex` (entire chapter, 7 pages)

**Issue:** STYLE.md targets 3--5 pages per chapter, with exceptions for Ch 5 (6--8) and Ch 6 (5--7). Chapter 7 runs 7 pages. The codebase tour covers 12 sections, some of which (7.10 Issue tracking, 7.11 Hard rules) substantially overlap with Chapter 8 (Sections 8.5 and 8.6 cover the same bd workflow and repository hygiene topics).

**Suggested fix:** Merge Sections 7.10--7.11 content into Chapter 8, which already covers both topics. This should bring Ch 7 down to ~5 pages. If full merging is too disruptive, cut the /plan /work /merge /fire descriptions from 7.10 (they appear verbatim in 8.5).

### M4. Chapters 7 and 8 overlap on bd workflow and repository hygiene

**Location:** `07-codebase.tex:149-178` (Sections 7.10--7.11) and `08-tooling.tex:69-96` (Sections 8.5--8.6)

**Issue:** The bd issue tracking workflow is described twice: once as a codebase-tour item (Section 7.10) and again as a tooling item (Section 8.5). The four slash commands (/plan, /work, /merge, /fire) are listed in both. Similarly, the "reference/ is frozen" rule and branch-naming conventions appear in both Section 7.11 and Section 8.6. A reader going front-to-back encounters the same material twice.

**Suggested fix:** Keep the full treatment in one chapter (Ch 8 is the natural home for workflow tooling) and reduce the other to a forward reference: "Issue tracking uses bd; \autoref{ch:tooling} describes the workflow."

### M5. Ch 5 double-label on chapter heading

**Location:** `05-persistent-homology.tex:2-3`

**Issue:** The chapter has two labels: `\label{ch:ph}` and `\label{ch:persistent-homology}`. Only `ch:persistent-homology` is referenced anywhere. The duplicate is harmless but violates the STYLE.md label convention (one label per element) and risks future confusion.

**Suggested fix:** Remove `\label{ch:ph}`. All existing autorefs already target `ch:persistent-homology`.

---

## MINOR

### m1. Ch 4, Section 4.3 (line 88): filtration nesting direction phrasing is inverted relative to Ch 5

**Location:** `04-graphs.tex:86-90` vs `05-persistent-homology.tex:111-130`

**Issue:** Chapter 4 describes the filtration as "ordered by increasing $\varepsilon$" where "every edge present at threshold $\varepsilon_2$ is also present at any lower threshold $\varepsilon_1 < \varepsilon_2$." This is correct for the Vietoris-Rips construction on distances (edges appear as epsilon grows). But two paragraphs earlier, Ch 4 line 93 says "As $\varepsilon$ rises, edges disappear, connected components fragment, and cycles break." This correctly describes the *threshold-on-weights* behavior (high epsilon = fewer edges above threshold), not the Rips filtration. The text glides between the two interpretations without flagging the duality. Chapter 5 consistently uses the Rips convention (edges appear as epsilon grows).

**Suggested fix:** Add a clarifying sentence at Ch 4 line 93, e.g.: "Note that rising $\varepsilon$ in the threshold-on-weights sense means fewer edges survive, which is the reverse of the Rips filtration convention where rising $\varepsilon$ admits more edges. The two are duals: our threshold $\varepsilon$ on weights corresponds to distance $1 - \varepsilon$ in the Rips frame."

### m2. Ch 6, Section 6.1 (line 7): "three benchmarks" claim lacks citation

**Location:** `06-pipeline.tex:7`

**Issue:** "A logistic regression over topological features achieves 93--98% accuracy across three benchmarks" is an empirical claim about Kushnareva's results but has no `\parencite{kushnareva2021}` on this specific sentence. The prior sentence does cite them, but STYLE.md's citation rule says "every factual or empirical claim cites a BibTeX entry."

**Suggested fix:** Add `\parencite{kushnareva2021}` after "three benchmarks."

### m3. Russian term gloss format inconsistency between Ch 1 and Ch 2

**Location:** `01-motivation.tex:6` vs `02-humanities.tex:18`

**Issue:** STYLE.md prescribes the pattern: "Cyrillic (transliteration, 'gloss')" with the Cyrillic in the original script. Chapter 1 uses `\cyr{...}` wrappers: `\cyr{голубой} (goluboj, 'light blue')`. Chapter 2 drops the `\cyr{}` wrapper entirely and writes bare Cyrillic: `синий (sinij, 'dark blue')`. Both render correctly in the PDF because `overview.tex` defines `\cyr` as a no-op and sets up a Cyrillic fallback font. But the inconsistency in source markup is confusing for future editing.

**Suggested fix:** Pick one convention for the whole document. Since `\cyr{}` is a no-op, the simplest fix is to remove the `\cyr{}` wrappers from Ch 1 so both chapters use bare Cyrillic. (Or keep `\cyr{}` everywhere for explicitness.)

### m4. Ch 5, Section 5.2: simplicial complex description says "closed under taking faces" without defining "face"

**Location:** `05-persistent-homology.tex:87-88`

**Issue:** The text says "A simplicial complex $K$ is a collection of simplices that is closed under taking faces: if a triangle belongs to $K$, then so do its three edges and three vertices." The example clarifies the concept, but "face" is never defined. For a reader meeting simplicial complexes for the first time (the target audience per STYLE.md), this is a gap.

**Suggested fix:** Add a parenthetical: "closed under taking faces (a face of a simplex is any simplex formed by a subset of its vertices)."

### m5. Ch 6, Section 6.5: "11--34 per language-domain pair" count is misleading

**Location:** `06-pipeline.tex:80`

**Issue:** The text says "Each canon term (11--34 per language-domain pair, sourced from published linguistics literature...)" but the range 11--34 spans across all nine cells. Within a single domain, the range is narrower (e.g., color is 11--12, emotion is 18--22, kinship is 27--34). Giving the full range without qualification makes it seem like any single cell could have 11 or 34 terms.

**Suggested fix:** Change to "Each canon term (11--12 for color, 18--22 for emotion, 27--34 for kinship, varying by language)" or just "(see Chapter 2 for counts)."

### m6. Unused bibliography entries

**Location:** `refs.bib`

**Issue:** Ten entries are defined but never cited: whorf1956, wu2020, rogers2020, kovaleva2019, greenberg1966, kronenfeld1996, apresjan2000, ekman1999, mann1947, golke2009. These were merged from per-chapter bib files during assembly. They inflate the bibliography without contributing.

**Suggested fix:** Remove uncited entries from `refs.bib`, or add citations where they belong. Candidates: `golke2009` (Leipzig Corpora) could be cited in Ch 6 Section 6.5 where Leipzig is mentioned; `mann1947` could be cited in Ch 6 Section 6.5 where Mann-Whitney tests are described.

### m7. Ch 5 notation table: $W_p$ symbol clashes with attention weight $W$

**Location:** `05-persistent-homology.tex:33` (Table 5.1)

**Issue:** The notation table defines $W$ as the attention weight matrix and $W_p$ as the $p$-Wasserstein distance. Both use capital $W$, which could confuse a reader scanning the table. STYLE.md does not prescribe a different symbol for Wasserstein distance, and the standard notation uses $W_p$, so this may be unavoidable.

**Suggested fix:** Add a note to the table row for $W_p$: "Not to be confused with the attention weight matrix $W$; context disambiguates." Or use $\mathcal{W}_p$ if the calligraphic form is acceptable.

### m8. Pipeline figure (Fig 6.1) is missing the "Cross-linguistic comparison" box

**Location:** `figures/pipeline.tex` renders on PDF p. 36

**Issue:** The pipeline figure in the PDF shows six boxes: Corpus, KWIC extraction, mBERT attention, Graph filtration, Persistent homology features, and a sixth box that is cut off at the right margin. The "Cross-linguistic comparison" text is partially visible but extends beyond the page boundary.

**Suggested fix:** Reduce `minimum width` in the `block` style from `6.5em` to `5.5em`, or reduce `node distance` from `1.0cm and 0.6cm` to `0.8cm and 0.5cm`, so all six boxes fit within the text width.

### m9. Ch 3, Section 3.7: forward reference says "chapter 4" in lowercase

**Location:** `03-attention.tex:71`

**Issue:** The `\autoref{ch:graphs}` renders as "chapter 4" with a lowercase "c" in the PDF. All other chapter autorefs in the document also render lowercase. The `report` class with `hyperref` defaults to lowercase for `\autoref` chapter names. This is consistent but may not be the desired style for a textbook.

**Suggested fix:** If capitalized "Chapter" is preferred, add to the preamble: `\renewcommand{\chapterautorefname}{Chapter}` and similarly for `\sectionautorefname{Section}`. This is a global style choice, not a per-chapter fix.

---

## Summary

| Severity | Count |
|----------|-------|
| BLOCKER  | 1     |
| MAJOR    | 5     |
| MINOR    | 9     |

**Top 3 issues to fix first:**

1. **B1** -- the symmetrization direction error in Ch 6 is a factual mistake in the mathematical description. It will confuse anyone checking the pipeline logic against Chapter 4.
2. **M1** -- the KWIC extraction status contradiction between Ch 6 and Ch 9 undermines the "honest accounting" that Ch 9 explicitly promises. One of them is wrong.
3. **M3/M4** -- the Ch 7 / Ch 8 overlap is the most visible multi-agent seam. Deduplicating the bd and hard-rules content would tighten both chapters.

**Things I noticed but did not flag:**

- Some chapters use em-dashes with spaces around them (`---`), others without. LaTeX renders both the same way, so this is purely a source-formatting preference.
- The Ekman basics list in Ch 2 names six emotions; the term-counts section says "the six basics plus contempt" for a base of 7. Contempt is a recognized seventh basic emotion in the Ekman literature (added in 1992), so this is correct but might benefit from a brief parenthetical in the six-item list.
- Chapter 8's prose voice is notably warmer and more self-aware than the other chapters ("nobody wants to write about," "the reason the build broke on a Tuesday"). This is a strength, not a problem -- it matches STYLE.md's "dialogical, rigorous-but-conversational" target better than the more measured tone of Ch 3 or Ch 4. But the temperature difference is noticeable.
- The document uses `\parencite` and `\textcite` correctly throughout -- parenthetical for claims, inline-by-name for narrative. This is well done.
- All three figures compile, render, and are referenced via `\autoref`. The persistence diagram figure (Fig 5.2) is particularly clean.
