from __future__ import annotations

from whsdsci.models.baseline_rate import BaselineLineMeanRateModel
from whsdsci.models.bayes_hier_poisson import BayesHierPoissonModel
from whsdsci.models.defense_two_step import DefenseAdjTwoStepModel
from whsdsci.models.elasticnet_rapm import ElasticNetRapmSoftplusModel
from whsdsci.models.hurdle_xg import HurdleXgModel
from whsdsci.models.lowrank_poisson_factor import LowRankPoissonFactorModel
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel
from whsdsci.models.poisson_glm_offset_reg import PoissonGlmOffsetRegModel
from whsdsci.models.ridge_rapm import RidgeRapmSoftplusModel
from whsdsci.models.two_stage_shots_xg import TwoStageShotsXgModel
from whsdsci.models.tweedie_glm import TweedieGlmRateModel


def get_model_builders(random_state: int = 0):
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


__all__ = ["get_model_builders"]
