import logging
import torch
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.estimation_algorithms.base_irt_algorithm import BaseIRTAlgorithm
from irtorch.estimation_algorithms import AE, VAE, MML
from irtorch.quantile_mv_normal import QuantileMVNormal
from irtorch.gaussian_mixture_model import GaussianMixtureModel
from irtorch.bit_scales import bit_scores_from_z, bit_score_gradients
from irtorch._internal_utils import linear_regression, dynamic_print, fix_missing_values
from irtorch.utils import one_hot_encode_test_data

logger = logging.getLogger("irtorch")

class IRTScorer:
    def __init__(self, model: BaseIRTModel, algorithm: BaseIRTAlgorithm):
        """
        Initializes the IRTScorer class with a given model and fitting algorithm.

        Parameters
        ----------
        model : BaseIRTModel
            BaseIRTModel object.
        algorithm : BaseIRTAlgorithm
            BaseIRTAlgorithm object.
        """
        self.model = model
        self.algorithm = algorithm
        self.latent_density = None
