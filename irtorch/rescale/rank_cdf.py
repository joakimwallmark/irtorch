import torch
from irtorch.rescale import Scale

class RankCDF(Scale):
    """
    Rank-based inverse CDF transformation of IRT theta scales.
    
    For each latent variable, finds the rank of each theta score in the population.
    For new data, finds the closest matching population ranks and uses the inverse CDF to 
    find the equivalents scores in the chosen distribution(s).

    Note that while this method is fast, it is heavily reliant on the input theta scores covering the entire range of the distribution(s).
    It is also not invertible, does not support gradient computation and the transformation is not unique one-to-one.
    
    Parameters
    ----------
    theta : torch.Tensor
        A large tensor of theta scores representing the population.
    distributions : list[torch.distributions.Distribution], optional
        The distributions to use for the transformation of each latent variable. If None, normal distributions are used.
    
    Examples
    --------
    >>> import irtorch
    >>> from irtorch.models import GradedResponse
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.rescale import RankCDF
    >>> data = irtorch.load_dataset.swedish_national_mathematics_1()
    >>> model = GradedResponse(data)
    >>> model.fit(train_data=data, algorithm=MML())
    >>> thetas = model.latent_scores(data)
    >>> # Create and RankCDF instancce and supply it to the model.
    >>> model.add_scale_transformation(RankCDF(thetas))
    >>> # Estimate thetas on the transformed scale
    >>> rescaled_thetas = model.latent_scores(data)
    >>> # Or alternatively by directly converting the old ones
    >>> rescaled_thetas = model.transform_theta(thetas)
    >>> # Plot the differences
    >>> model.plot.latent_score_distribution(thetas).show()
    >>> model.plot.latent_score_distribution(rescaled_thetas).show()
    >>> # Plot an item on the transformed scale
    >>> model.plot.item_probabilities(1).show()
    """
    def __init__(self, theta: torch.Tensor, distributions: list[torch.distributions.Distribution] | None = None):
        super().__init__(invertible=False)
        self.n_samples, latent_variables = theta.shape
        if distributions is None:
            self.distributions = [
                torch.distributions.Normal(0, 1) for _ in range(latent_variables)
            ]
        else:
            if theta.shape[1] != len(distributions) and len(distributions) != 1:
                raise ValueError("The number of distributions must be one or equal to the number of latent variables.")
            if len(distributions) == 1:
                self.distributions = [distributions[0] for _ in range(latent_variables)]
            else:
                self.distributions = distributions

        self.unique_vals_list = []
        self.avg_ranks_list = []

        for feature in range(latent_variables):
            sorted_data = torch.sort(theta[:, feature]).values
            # Generate rank indices for the sorted data
            rank = torch.arange(1, self.n_samples + 1, dtype=torch.float32, device=theta.device)
            # Handle ties: Compute unique values and assign average rank for ties
            unique_vals, inverse_indices = torch.unique(sorted_data, return_inverse=True, sorted=True)
            # Sum ranks for the same unique values (i.e., tie groups) and count the number of occurrences per group
            rank_sum = torch.zeros_like(unique_vals)
            rank_count = torch.zeros_like(unique_vals)
            rank_sum = rank_sum.scatter_add(0, inverse_indices, rank)
            rank_count = rank_count.scatter_add(0, inverse_indices, torch.ones_like(rank))
            # Compute the average rank for each unique value
            avg_ranks = rank_sum / rank_count
            self.avg_ranks_list.append(avg_ranks)
            self.unique_vals_list.append(unique_vals)

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.
        
        Returns
        -------
        torch.Tensor
            A 2D tensor containing the transformed theta scores.
        """
        _, latent_variables = theta.shape
        transformed_data = torch.zeros_like(theta, dtype=torch.float32)

        for latent_variable in range(latent_variables):
            unique_vals = self.unique_vals_list[latent_variable].to(theta.device)
            avg_ranks = self.avg_ranks_list[latent_variable].to(theta.device)
            col_theta = theta[:, latent_variable].contiguous()
            
            # Find insertion points
            idx = torch.searchsorted(unique_vals, col_theta)
            
            # We need to compare index `idx` and `idx-1`.
            # Clip idx to be valid indices for `unique_vals`
            idx_right = idx.clamp(max=len(unique_vals) - 1)
            idx_left = (idx - 1).clamp(min=0)
            
            dist_right = torch.abs(unique_vals[idx_right] - col_theta)
            dist_left = torch.abs(unique_vals[idx_left] - col_theta)
            
            # Choose indices where left is closer
            use_left = dist_left < dist_right
            final_idx = torch.where(use_left, idx_left, idx_right)
            
            new_ranks = avg_ranks[final_idx]

            # Normalize the ranks to (0, 1) and apply the inverse CDF
            rank_normalized = new_ranks / (self.n_samples + 1)
            transformed_data[:, latent_variable] = \
                self.distributions[latent_variable].icdf(rank_normalized.cpu()).to(theta.device)

        return transformed_data

    def inverse(self, transformed_theta):
        raise NotImplementedError("RankCDF is not invertible.")

    def jacobian(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of scale scores with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        raise NotImplementedError("Gradients are not available for RankCDF scale transformations.")
