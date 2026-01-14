# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 16:54:14 2025

@author: Allison Walker
"""
# adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.compose import ColumnTransformer
import random
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import tqdm
from torch.utils.data import Dataset, DataLoader


def pairwise_distances(x, eps=1e-9):
    """
    Compute full pairwise Euclidean distance matrix for input x.
    x: (N, d) tensor
    Returns: (N, N) distance matrix
    """
    # x2: squared norms
    x2 = (x**2).sum(dim=1, keepdim=True)  # (N,1)

    # Dot product matrix
    xy = x @ x.t()  # (N,N)

    # Compute squared distances: ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    dist_sq = x2 + x2.t() - 2 * xy
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Euclidean distances
    return torch.sqrt(dist_sq + eps)


def pairwise_distance_vector(x):
    """
    Returns pairwise Euclidean distances as a vector of size N*(N-1)/2
    (upper triangular part).
    """
    D = pairwise_distances(x)
    i, j = torch.triu_indices(D.size(0), D.size(0), offset=1)
    return D[i, j]


def cosine_similarity_matrix(x, eps=1e-8):
    """
    Computes full pairwise cosine similarity matrix for rows of x.
    x: (N, d)
    Returns: (N, N) cosine similarity matrix
    """
    # Normalize each row to unit norm
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)  # avoid division by zero

    # Cosine sim = dot product of normalized vectors
    sim = x_norm @ x_norm.t()  # (N, N)

    # Clamp for numerical stability (cosine should be in [-1, 1])
    sim = torch.clamp(sim, -1.0, 1.0)
    return sim


def cosine_distance_matrix(x, eps=1e-8):
    """
    Computes pairwise cosine distance = 1 - cosine_similarity.
    """
    sim = cosine_similarity_matrix(x, eps=eps)
    dist = 1.0 - sim
    return dist


def pairwise_cosine_vector(x):
    """
    Returns pairwise Euclidean distances as a vector of size N*(N-1)/2
    (upper triangular part).
    """
    D = cosine_distance_matrix(x)
    i, j = torch.triu_indices(D.size(0), D.size(0), offset=1)
    return D[i, j]
