"""Modern-transformers shim around reference/grab_weights.py.

The original Kushnareva code targets transformers 4.3.0, which used
`tokenizer.batch_encode_plus(..., pad_to_max_length=True)`. transformers 5.x
removed `batch_encode_plus` and renamed `pad_to_max_length` to
`padding="max_length"`. This module replaces just `grab_attention_weights`
with the modern API. Everything else (text_preprocessing) is re-exported
unchanged from reference/.

Notebooks in replication/notebooks/ import from this module instead of
`grab_weights` directly. reference/ stays frozen.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REFERENCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference"))
if _REFERENCE not in sys.path:
    sys.path.insert(0, _REFERENCE)

from grab_weights import text_preprocessing  # noqa: E402, F401


def grab_attention_weights(model, tokenizer, sentences, MAX_LEN, device="cuda:0"):
    """Tokenize a batch and return attention tensors.

    Returns: numpy array shaped (n_layers, batch, n_heads, MAX_LEN, MAX_LEN),
    dtype float16. Behavior matches reference/grab_weights.py:grab_attention_weights
    apart from using the modern transformers tokenizer API.
    """
    inputs = tokenizer(
        [text_preprocessing(s) for s in sentences],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    attention = model(input_ids, attention_mask, token_type_ids)["attentions"]
    return np.asarray(
        [layer.cpu().detach().numpy() for layer in attention], dtype=np.float16
    )


def grab_attention_and_embeddings(
    model, tokenizer, sentences, MAX_LEN, device="cuda:0"
):
    """Run mBERT once, return both attention and final-layer embeddings.

    Requirements:
      - model loaded with output_attentions=True AND output_hidden_states=True
      - tokenizer must be a fast tokenizer (BertTokenizerFast or similar) so
        that BatchEncoding.word_ids(batch_index=...) is available.
        A slow tokenizer raises ValueError.

    Returns
    -------
    attention : np.ndarray, dtype float16
        Shape (n_layers, batch, n_heads, MAX_LEN, MAX_LEN). Same as
        grab_attention_weights — bit-for-bit identical for the same input.
    embeddings : np.ndarray, dtype float16
        Shape (batch, MAX_LEN, hidden_size). Final-layer hidden state
        (outputs['hidden_states'][-1]).
    batch_word_ids : list[list[Optional[int]]]
        Length batch; each inner list length MAX_LEN. Each entry is the
        whitespace-word index the wordpiece came from, or None for
        [CLS]/[SEP]/[PAD]. Used downstream to align canon-term `target_idx`
        (whitespace space) to wordpiece span.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"grab_attention_and_embeddings requires a fast tokenizer "
            f"(e.g. BertTokenizerFast); got {type(tokenizer)}"
        )

    encoded = tokenizer(
        [text_preprocessing(s) for s in sentences],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask, token_type_ids)

    attention_np = np.asarray(
        [layer.cpu().detach().numpy() for layer in outputs["attentions"]],
        dtype=np.float16,
    )

    last_hidden = outputs["hidden_states"][-1]
    embeddings_np = last_hidden.cpu().detach().numpy().astype(np.float16)

    batch_word_ids = [encoded.word_ids(batch_index=i) for i in range(len(sentences))]

    return attention_np, embeddings_np, batch_word_ids
