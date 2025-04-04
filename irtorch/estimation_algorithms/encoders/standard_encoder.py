import logging
from torch import nn
from irtorch.estimation_algorithms.encoders import BaseEncoder

logger = logging.getLogger("irtorch")

class StandardEncoder(BaseEncoder):
    def __init__(
        self,
        input_dim: int,
        latent_variables: int,
        hidden_dim: list[int],
        batch_normalization: bool = True,
        nonlinear=nn.ReLU(),
    ):
        super().__init__(input_dim, latent_variables)
        self.layers = nn.Sequential()
        # Input layer
        self.layers.add_module("linear0", nn.Linear(input_dim, hidden_dim[0]))
        if batch_normalization:
            self.layers.add_module("batchnorm0", nn.BatchNorm1d(hidden_dim[0]))
        self.layers.add_module("nonlinear0", nonlinear)

        # Hidden layers
        for i in range(1, len(hidden_dim)):
            self.layers.add_module(
                f"linear{i}", nn.Linear(hidden_dim[i - 1], hidden_dim[i])
            )
            if batch_normalization:
                self.layers.add_module(f"batchnorm0{i}", nn.BatchNorm1d(hidden_dim[i]))
            self.layers.add_module(f"nonlinear{i}", nonlinear)

        # output layer
        self.layers.add_module(
            f"linear{len(hidden_dim)+1}", nn.Linear(hidden_dim[-1], latent_variables)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.layers[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.layers[-1].bias, mean=0., std=0.001)

    def forward(self, x):
        return self.layers(x)
