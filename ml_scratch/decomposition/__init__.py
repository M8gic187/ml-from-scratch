"""Dimensionality reduction algorithms."""

from .ica import FastICA
from .lda import LinearDiscriminantAnalysis
from .pca import PCA
from .tsne import TSNE

__all__ = ["FastICA", "LinearDiscriminantAnalysis", "PCA", "TSNE"]
