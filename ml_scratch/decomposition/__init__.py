"""Dimensionality reduction algorithms."""

from .ica import FastICA
from .lda import LinearDiscriminantAnalysis
from .nmf import NMF
from .pca import PCA
from .tsne import TSNE

__all__ = ["FastICA", "LinearDiscriminantAnalysis", "NMF", "PCA", "TSNE"]
