"""
baselines — static embedding baseline modules for ph-project.

Sub-baselines:
  A. Language-specific descriptive (fastText CC-300)
  B. Language-specific topological (fastText + persistent homology)
  C. Cross-lingual aligned (MUSE supervised alignment)
"""

# Single source of truth for the supported language set.
# Gates the lang= argument in vectors.py (load_fasttext) and
# distances.py (extract_term_vectors / HEAD_POSITION).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "ru", "es"})
