from __future__ import annotations

from whsdsci.models.tree_poisson_best import TreePoissonBestModel, get_best_base_model_builders


def get_model_builders(random_state: int = 0):
    """Base-model registry retained for the frozen best system."""
    return get_best_base_model_builders(random_state=random_state)


__all__ = ["TreePoissonBestModel", "get_best_base_model_builders", "get_model_builders"]

