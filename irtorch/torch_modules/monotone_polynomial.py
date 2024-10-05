import torch
from torch import nn
import torch.nn.functional as F

class MonotonePolynomialModule(nn.Module):
    """
    A polynomial with monotonicity constraints.

    Parameters
    ----------
    degree: int
        Degree of the polynomial. Needs to be an uneven number.
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    intercept: bool
        Whether to include an intercept term. (Default: False)
    relationship_matrix : torch.Tensor, optional
        A boolean tensor of shape (in_features, out_features,) that determines which inputs are related to which outputs. Typically used for IRT models to remove relationships between some items or item cateogires and latent traits. (Default: None)
    negative_relationships : bool, optional
        Whether to allow for negative relationships. (Default: False)
    shared_directions : int, optional
        Only when negative_relationships is true. Number of out_features with shared relationship directions. out_features needs to be divisible with this. (Default: 1)
    """
    def __init__(
        self,
        degree: int,
        in_features: int = 1,
        out_features: int = 1,
        intercept: int = False,
        relationship_matrix: torch.Tensor = None,
        negative_relationships: bool = False,
        shared_directions: int = 1
    ) -> None:
        super().__init__()
        if degree % 2 == 0:
            raise ValueError("Degree must be an uneven number.")
        self.k = (degree - 1) // 2
        self.input_dim = in_features
        self.output_dim = out_features
        self.relationship_matrix = relationship_matrix
        self.negative_relationships = negative_relationships
        self.shared_directions = shared_directions

        self.omega = nn.Parameter(torch.zeros(1, in_features, out_features, requires_grad=True))
        self.alpha = nn.Parameter(torch.zeros(self.k, in_features, out_features, requires_grad=True))
        self.tau = nn.Parameter(torch.full((self.k, in_features, out_features), -5.0, requires_grad=True))
        if intercept:
            self.intercept = nn.Parameter(torch.zeros(out_features, requires_grad=True))
        else:
            self.register_buffer('intercept', None)

        if negative_relationships:
            if shared_directions == 0:
                raise ValueError("shared_directions must be greater than 0.")
            if out_features % shared_directions != 0:
                raise ValueError("out_features must be divisible by shared_directions.")
            self.directions = nn.Parameter(torch.zeros(in_features, int(out_features / shared_directions), requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp_tau = F.softplus(self.tau)
        
        b = F.softplus(self.omega)
        for i in range(self.k):
            matrix = torch.zeros((2*(i+1)+1, 2*(i+1)-1, self.input_dim, self.output_dim), device=x.device)
            range_indices = torch.arange(2*(i+1)-1, device=x.device)
            matrix[range_indices, range_indices, :, :] = 1
            matrix[range_indices + 1, range_indices, :, :] = -2 * self.alpha[i]
            matrix[range_indices + 2, range_indices, :, :] = self.alpha[i] ** 2 + sp_tau[i]
            b = torch.einsum('abio,bio->aio', matrix, b) / (i + 1)
        
        if self.negative_relationships:
            b.multiply_(self.directions.repeat_interleave(self.shared_directions, dim=1))

        # remove relationship between some items and latent variables
        if self.relationship_matrix is not None:
            b[:, ~self.relationship_matrix] = 0.0
        x_powers = x.unsqueeze(2) ** torch.arange(1, 2*self.k+2, device=x.device)
        # x_powers dimensions: (batch, input_dim, degree)
        # b dimensions: (degree, input_dim, output_dim)
        result = torch.einsum('abc,cbd->ad', x_powers, b)
        if self.intercept is not None:
            result += self.intercept
        return result
    
    @torch.inference_mode()
    def get_polynomial_coefficients(self) -> torch.Tensor:
        """
        Returns the polynomial coefficients.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of tensors containing the coefficients of the polynomial with dimensions (degree, input_dim, output_dim), the second tensor contains the intercept if it exists and None otherwise.
        """
        sp_tau = F.softplus(self.tau)
        
        b = F.softplus(self.omega)
        for i in range(self.k):
            matrix = torch.zeros((2*(i+1)+1, 2*(i+1)-1, self.input_dim, self.output_dim))
            range_indices = torch.arange(2*(i+1)-1)
            matrix[range_indices, range_indices, :, :] = 1
            matrix[range_indices + 1, range_indices, :, :] = -2 * self.alpha[i]
            matrix[range_indices + 2, range_indices, :, :] = self.alpha[i] ** 2 + sp_tau[i]
            b = torch.einsum('abio,bio->aio', matrix, b) / (i + 1)

        if self.negative_relationships:
            b.multiply_(self.directions.repeat_interleave(self.shared_directions, dim=1))

        # remove relationship between some items and latent variables
        if self.relationship_matrix is not None:
            b[:, ~self.relationship_matrix] = 0.0

        return b, self.intercept
