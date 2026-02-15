from __future__ import annotations

from whsdsci.models.baseline_rate import BaselineLineMeanRateModel
from whsdsci.models.bayes_hier_poisson import BayesHierPoissonModel
from whsdsci.models.defense_two_step import DefenseAdjTwoStepModel
from whsdsci.models.elasticnet_rapm import ElasticNetRapmSoftplusModel
from whsdsci.models.ensemble_convex import EnsembleConvexModel
from whsdsci.models.ensemble_foldavg import EnsembleFoldAvgModel
from whsdsci.models.ensemble_mean import EnsembleMeanModel
from whsdsci.models.ensemble_moe import EnsembleMoEModel
from whsdsci.models.ensemble_stack_poisson import EnsembleStackPoissonModel
from whsdsci.models.hurdle_xg import HurdleXgModel
from whsdsci.models.lowrank_poisson_factor import LowRankPoissonFactorModel
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel
from whsdsci.models.poisson_glm_offset_reg import PoissonGlmOffsetRegModel
from whsdsci.models.ridge_rapm import RidgeRapmSoftplusModel
from whsdsci.models.tweedie_glm import TweedieGlmRateModel
from whsdsci.models.two_stage_shots_xg import TwoStageShotsXgModel


def get_model_builders(random_state: int = 0):
    """Base (non-ensemble) model registry."""
    return {
        "BASELINE_LINE_MEAN_RATE": lambda: BaselineLineMeanRateModel(random_state=random_state),
        "DEFENSE_ADJ_TWO_STEP": lambda: DefenseAdjTwoStepModel(random_state=random_state),
        "RIDGE_RAPM_RATE_SOFTPLUS": lambda: RidgeRapmSoftplusModel(random_state=random_state),
        "ELASTICNET_RAPM_RATE_SOFTPLUS": lambda: ElasticNetRapmSoftplusModel(random_state=random_state),
        "POISSON_GLM_OFFSET": lambda: PoissonGlmOffsetModel(random_state=random_state),
        "POISSON_GLM_OFFSET_REG": lambda: PoissonGlmOffsetRegModel(random_state=random_state),
        "TWEEDIE_GLM_RATE": lambda: TweedieGlmRateModel(random_state=random_state),
        "TWO_STAGE_SHOTS_XG": lambda: TwoStageShotsXgModel(random_state=random_state),
        "HURDLE_XG": lambda: HurdleXgModel(random_state=random_state),
        "LOWRANK_POISSON_FACTOR": lambda: LowRankPoissonFactorModel(random_state=random_state),
        "BAYES_HIER_POISSON_OFFSET": lambda: BayesHierPoissonModel(random_state=random_state),
    }


def get_ensemble_model_builders(
    random_state: int,
    base_model_builders: dict,
    base_model_names: list[str],
):
    names = [n for n in base_model_names if n in base_model_builders]
    if not names:
        return {}

    foldavg_base = "POISSON_GLM_OFFSET" if "POISSON_GLM_OFFSET" in names else names[0]

    return {
        "ENSEMBLE_MEAN_TOPK": lambda: EnsembleMeanModel(
            random_state=random_state,
            base_model_builders=base_model_builders,
            base_model_names=names,
            inner_splits=3,
        ),
        "ENSEMBLE_CONVEX_TOPK": lambda: EnsembleConvexModel(
            random_state=random_state,
            base_model_builders=base_model_builders,
            base_model_names=names,
            inner_splits=3,
        ),
        "ENSEMBLE_STACK_POISSON": lambda: EnsembleStackPoissonModel(
            random_state=random_state,
            base_model_builders=base_model_builders,
            base_model_names=names,
            inner_splits=3,
        ),
        "ENSEMBLE_MOE_2EXPERT": lambda: EnsembleMoEModel(
            random_state=random_state,
            base_model_builders=base_model_builders,
            base_model_names=names[: max(2, min(3, len(names)))],
            inner_splits=3,
        ),
        "ENSEMBLE_FOLDAVG": lambda: EnsembleFoldAvgModel(
            random_state=random_state,
            base_model_builders=base_model_builders,
            base_model_names=names,
            base_model_name=foldavg_base,
            n_folds=3,
            inner_splits=3,
        ),
    }


__all__ = ["get_model_builders", "get_ensemble_model_builders"]
