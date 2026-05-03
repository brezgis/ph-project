"""
phase1_kwic — KWIC extraction pipeline for ph-project Phase 1.

This package implements canon-term loading, per-language lemma matching, and
the sentence extraction pipeline that produces the per-(lang, domain) CSVs
described in data/kwic/SCHEMA.md.

Exports
-------
SUPPORTED_LANGUAGES
    Frozenset of BCP-47-style language codes supported by this project.
    Re-exported from baselines to keep a single source of truth; do NOT
    duplicate the definition here.

DOMAINS
    Frozenset of semantic domain names used in the canon-term YAMLs.
"""

import baselines

# Single source of truth: defined in baselines/__init__.py.
# Import and re-export so callers can write:
#     from phase1_kwic import SUPPORTED_LANGUAGES, DOMAINS
# without reaching into baselines directly.
SUPPORTED_LANGUAGES = baselines.SUPPORTED_LANGUAGES

# The three semantic domains covered by the Phase 1 KWIC extraction.
DOMAINS: frozenset[str] = frozenset({"color", "emotion", "kinship"})
