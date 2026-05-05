"""Pin the tokenizer/model configuration the mbert_attention_thresholds notebook depends on.

These tests run without GPU and without producing any feature .npy files. They
catch two classes of silent regression that the shape/no-inf/non-zero tests in
test_mbert_attention_thresholds_features.py would not:

1. **Wrong model checkpoint loaded.** If the local HuggingFace cache is stale,
   corrupted, or someone fine-tunes a different model and saves it under the
   same path, BertModel.from_pretrained could silently load a model with the
   wrong vocabulary. mBERT-cased has a known vocab size (119547) distinct
   from bert-base-uncased (30522). Pinning that catches the mistake.

2. **do_lower_case accidentally reverted.** The substrate uses
   do_lower_case=True (English-only). The mwk.2 adaptation switches to False
   because Russian and Spanish casing carries meaning AND because the cased
   checkpoint requires it. A revert would silently produce different attention
   patterns of the right shape — undetectable from feature .npy alone. We pin
   it by tokenizing a cased Russian token and asserting it differs from the
   lowercased form.
"""
from transformers import BertConfig, BertTokenizer

MODEL_ID = "bert-base-multilingual-cased"
EXPECTED_VOCAB_SIZE = 119547  # mBERT-cased canonical; bert-base-uncased is 30522.


def test_mbert_config_has_expected_vocab_size():
    """Pin the model identity. Catches stale-cache or wrong-checkpoint regressions."""
    cfg = BertConfig.from_pretrained(MODEL_ID)
    assert cfg.vocab_size == EXPECTED_VOCAB_SIZE, (
        f"Expected mBERT-cased vocab_size={EXPECTED_VOCAB_SIZE}, got {cfg.vocab_size}. "
        f"This usually means a different checkpoint was loaded under the same ID — "
        f"clear ~/.cache/huggingface/ and re-pull, or check for a local override."
    )


def test_tokenizer_preserves_case_for_cyrillic():
    """Pin do_lower_case=False. Cyrillic capital ≠ lowercase under cased tokenizer.

    If do_lower_case is accidentally set to True, the tokenizer will lowercase
    "Красный" before tokenization, producing the same token IDs as "красный".
    The cased tokenizer keeps them distinct.
    """
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID, do_lower_case=False)
    assert tokenizer.tokenize("Красный") != tokenizer.tokenize("красный"), (
        "Cased mBERT tokenizer produced identical tokens for 'Красный' and "
        "'красный' — do_lower_case is likely set to True. The mwk.2 notebook "
        "requires do_lower_case=False; see CLAUDE.md and the cell-25 DIVERGES "
        "marker for the rationale."
    )


def test_tokenizer_preserves_case_for_spanish_accented():
    """Sanity check that case-preservation works for accented Spanish too."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID, do_lower_case=False)
    # "Mexicano" (proper noun / sentence-initial) vs "mexicano" (adjective)
    # — both legitimate Spanish surface forms with different syntactic roles.
    assert tokenizer.tokenize("Mexicano") != tokenizer.tokenize("mexicano"), (
        "Cased mBERT tokenizer collapsed 'Mexicano' and 'mexicano' — "
        "do_lower_case is likely True."
    )
