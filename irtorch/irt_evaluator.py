import logging
import torch
from torch.distributions import MultivariateNormal
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm, VAE, AE, MML
from irtorch.quantile_mv_normal import QuantileMVNormal
from irtorch.gaussian_mixture_model import GaussianMixtureModel
from irtorch.irt_scorer import IRTScorer
from irtorch._internal_utils import (
    fix_missing_values,
    impute_missing,
    conditional_score_distribution,
    sum_incorrect_probabilities,
)

logger = logging.getLogger("irtorch")

class IRTEvaluator:
    def __init__(self, model: BaseIRTModel, algorithm: BaseIRTAlgorithm, scorer: IRTScorer):
        """
        Initializes the IRTEvaluator class.

        Parameters
        ----------
        model : BaseIRTModel
            BaseIRTModel object.
        algorithm : BaseIRTAlgorithm
            BaseIRTAlgorithm object.
        scorer : IRTScorer
            IRTScorer object used to obtain latent variable scores.
        """
        self.model = model
        self.algorithm = algorithm
        self.scorer = scorer
