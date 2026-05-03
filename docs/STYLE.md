# Style guide for the textbook overview

This document governs tone, terminology, and formatting across all chapters. When in doubt, follow this guide rather than inventing local conventions.

## Voice

Dialogical, rigorous-but-conversational. Pose questions and answer them. Build intuition before formalism.

First-person plural: "we predict", "we choose mBERT because", "our pipeline extracts". Not third-person ("the author predicts"); not passive ("it is predicted that").

Treat the reader as someone deep in linguistics meeting the topology fresh. They know morphological paradigms and semantic fields; they do not (yet) know what a simplicial complex is. Write as if you are explaining to a brilliant colleague who just asked "so what is persistent homology, exactly?" over coffee.

Example of the target register:

> Why mBERT rather than a monolingual model for each language? Because we need representations that share a vector space. If Russian and English live in separate embedding spaces, "closer" and "farther" across languages has no meaning. mBERT gives us a single 768-dimensional space trained on 104 languages simultaneously — distances within it are cross-linguistically comparable by construction.

## Banned constructions (with examples)

These weaken prose. Replace them with direct statement or restructure.

| Banned | Why | Instead |
|--------|-----|---------|
| "It's worth noting that X" | Throat-clearing; just state X. | "X." |
| "Furthermore" / "Moreover" / "Additionally" | Filler connectives. | Start the new sentence directly, or use a paragraph break. |
| "Let me explain" / "Let us now turn to" | Meta-narration about the text. | Just begin the explanation. |
| Three-bullet summary at end of section | Patronizing recap. | Let the prose conclude naturally. |
| "may potentially" / "could possibly" / "arguably" | Hedge stacking. | Commit ("we expect") or omit the claim. |
| "In conclusion" / "To summarize" | Telegraphs structure the reader can see. | Write a concluding sentence that concludes. |
| "It is important to note" | If it's important, it's in the prose. | State the thing. |
| "This is because" (sentence-initial) | Weak causality framing. | Fold the cause into the preceding sentence with "because" or restructure. |

## Citation rule

Every factual or empirical claim cites a BibTeX entry. No "[citation needed]" or unsourced hedging.

- Parenthetical citation: `\parencite{berlin1969}` renders as "(Berlin & Kay, 1969)"
- Inline-by-name: `\textcite{winawer2007}` renders as "Winawer et al. (2007)"

When adding a citation, scan `refs.bib` for an existing match before appending. New entries should follow the style of the existing thematic groups (color, emotion, kinship, attention, topology) and use real DOIs / ACL anthology IDs / publisher info — not placeholders.

## Math typesetting

- Inline math: `$...$` (never `\(...\)`)
- Display math: `\[...\]` or `align` environment (never `$$...$$`)
- Hypotheses: `$H_0$` and `$H_1$` in math mode, never Unicode subscripts
- Filtration parameter: `$\varepsilon$` consistently (not `$\epsilon$`)
- Distance: lowercase $d$ (e.g., `$d(u, v)$`)
- Attention weight / similarity: capital $W$ (e.g., `$W_{ij}$`)
- Betti numbers: `$\beta_0$`, `$\beta_1$` (lowercase beta, numeric subscript)
- Persistence pairs: `$(b_i, d_i)$` for birth/death (avoid ambiguity with distance $d$ by context or using `$d_i^{\text{death}}$` if needed)

## Terminology consistency

Use these terms consistently across all chapters. On first use in each chapter, define or gloss the term; thereafter use the short form.

| Concept | Preferred term | Avoid |
|---------|---------------|-------|
| The threshold parameter in filtration | "filtration parameter $\varepsilon$" or "threshold $\varepsilon$" | "epsilon", "eps", "distance cutoff" |
| A (birth, death) pair | "persistence pair" or "(birth, death) pair" | "barcode bar", "interval" (alone) |
| The multiset of persistence pairs | "barcode" | "barcode diagram" |
| The 2D plot of (birth, death) points | "persistence diagram" | "PD plot", "birth-death plot" |
| Identifying an attention head | "layer $\ell$ head $h$" (e.g., "layer 4 head 7") | "L4H7", "head 4-7" |
| Connected components (H_0) | "$H_0$ features" or "connected components" | "zero-dimensional holes" |
| Loops (H_1) | "$H_1$ features" or "loops" / "cycles" | "one-dimensional holes" |

**Russian terms:** Cyrillic with inline gloss on first use per chapter.
- Pattern: "тоска (toska, 'longing')" — then either form thereafter.

**Spanish terms:** Original orthography, glossed inline on first use per chapter.
- Pattern: "duende ('a dark, telluric force')" — then the Spanish form alone.

## Section titles

Prose, not numeric. Write `\section{Color}`, not `\section{2.1 Color}`. LaTeX handles numbering via the `report` class. Chapter titles are set in the stub files; section titles within chapters follow the same principle.

Good: `\section{Why persistent homology?}`
Bad: `\section{5.1: Motivation for PH}`

## Notation

A consolidated notation table belongs in Chapter 5 (persistent homology). Other chapters introduce notation inline as needed and may reference Chapter 5's table via `\autoref`.

The notation table should cover at minimum: $\varepsilon$, $d$, $W$, $\beta_0$, $\beta_1$, $H_0$, $H_1$, $(b, d)$ pairs.

## Cross-references

Use `\autoref` for all internal references. It generates "Chapter 5", "Figure 3.1", "Section 2.3" automatically.

Label conventions:
- One label per chapter, full name (e.g., `\label{ch:persistent-homology}`, not `\label{ch:ph}`). Aliases drift; pick the long form and stick with it.
- Sections: `\label{sec:russian-blue}`, `\label{sec:filtration}`
- Figures: `\label{fig:pipeline}`, `\label{fig:persistence-diagram}`
- Tables: `\label{tab:notation}`, `\label{tab:term-counts}`
- Equations: `\label{eq:distance}`, `\label{eq:betti}`

Cyrillic and Spanish accents render natively under LuaLaTeX with the
luaotfload Cyrillic fallback configured in `overview.tex`. Write Cyrillic
inline as bare Unicode — no `\cyr{...}` wrapper needed.

## Length

Chapters target 3--5 pages of body text (roughly 1500--2500 words).

Exceptions:
- Chapter 5 (persistent homology): 6--8 pages. This is where the mathematical core lives; it needs space to build intuition properly.
- Chapter 6 (pipeline and adaptation): 5--7 pages. Two pipelines to explain (Kushnareva's original and our adaptation) plus the mapping between them.

These are targets, not hard limits. A chapter that says what it needs to say in 3 pages is better than one padded to 5.
