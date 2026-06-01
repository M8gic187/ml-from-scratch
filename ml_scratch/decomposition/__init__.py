"""Dimensionality reduction algorithms."""

from .lda import LinearDiscriminantAnalysis
from .pca import PCA
from .tsne import TSNE

__all__ = ["LinearDiscriminantAnalysis", "PCA", "TSNE"]
