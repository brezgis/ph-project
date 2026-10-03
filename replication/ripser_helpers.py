"""Module-level worker functions for the ripser barcode subprocess.

Lives in a module (not in the notebook) because the ripser/templates
notebook uses spawn-context multiprocessing — spawn launches a fresh
Python interpreter that imports the worker by qualified name, which
cell-defined functions in Jupyter's __main__ cannot satisfy.
"""
from __future__ import annotations

import itertools

import ripser_count


def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
    queue.close()


def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = ripser_count.get_barcodes(
            matricies, ntokens_array, dim, lower_bound, (layer, head)
        )
    return barcodes
