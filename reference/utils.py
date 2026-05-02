import numpy as np

def cutoff_matrix(matrix, ntokens):
    """Return normalized submatrix of first n_tokens"""
    matrix = matrix[:ntokens, :ntokens]         # trim attn matrix to real token count, slice off padding
    matrix /= matrix.sum(axis=1, keepdims=True) # row normalize such that rows sum to 1
    return matrix