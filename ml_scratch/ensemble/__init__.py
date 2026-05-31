"""Ensemble learning algorithms."""

from .adaboost import AdaBoostClassifier
from .extra_trees import ExtraTreesClassifier
from .gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from .isolation_forest import IsolationForest
from .random_forest import RandomForestClassifier

__all__ = [
    "AdaBoostClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "IsolationForest",
    "RandomForestClassifier",
]
