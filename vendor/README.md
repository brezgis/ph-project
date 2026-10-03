# vendor/

## tda4atd

[`danchern97/tda4atd`](https://github.com/danchern97/tda4atd), the original code for
Kushnareva et al. (2021), *Artificial Text Detection via Examining the Topology of Attention
Maps* (EMNLP), included as a git submodule pinned to commit `aabf0a6`. It's used unmodified:
the attention notebooks import `grab_weights`, `ripser_count` and `stats_count` from it, and
`attention_topology/grab_weights_compat.py` patches the one function that no longer runs on
current `transformers`. The upstream repo has no license file, so it's linked rather than
copied, and this repository's MIT license doesn't cover it.

Fetch it with:

```bash
git submodule update --init
```

### Reading notes

Notes I made while working through the code:

- `grab_weights.grab_attention_weights`
  1. tokenizes a batch of sentences
  2. runs BERT with `output_attentions=True`
  3. returns a numpy array shaped layers × samples × heads × tokens × tokens
- `grab_weights.grab_weights_for_all` is a convenience wrapper: it saves/caches a single
  layer + head's matrices to `.npy` and loads previously computed weights.
- `utils.cutoff_matrix`
  - `matrix[:ntokens, :ntokens]` trims the attention matrix to the real token count, slicing
    off padding
  - `matrix /= matrix.sum(axis=1, keepdims=True)` row-normalizes so each row sums to 1
