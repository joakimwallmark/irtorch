import torch
from scipy.stats import norm

class RankInverseNormalTransformer:
    def __init__(self):
        self.unique_vals_list = []
        self.avg_ranks_list = []
        self.n_samples = 0

    def fit(self, data):
        """Construct the transformation based on the original data."""
        n_samples, n_features = data.shape
        self.n_samples = n_samples

        # Clear the stored transformation data
        self.unique_vals_list = []
        self.avg_ranks_list = []

        for feature in range(n_features):
            # Sort the data for each feature and get unique values
            sorted_data = torch.sort(data[:, feature]).values

            # Generate rank indices for the sorted data
            rank = torch.arange(1, n_samples + 1, dtype=torch.float32)

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

    def transform(self, data):
        """Transform new data to the normal scale based on the fitted transformation."""
        n_samples, n_features = data.shape
        transformed_data = torch.zeros_like(data, dtype=torch.float32)

        for feature in range(n_features):
            unique_vals = self.unique_vals_list[feature]
            avg_ranks = self.avg_ranks_list[feature]

            # Step 1: For each value in the new data, find the corresponding rank from the original data
            new_ranks = torch.zeros(n_samples)
            for i in range(n_samples):
                # Find the closest matching unique value (use nearest value approach)
                closest_idx = torch.abs(unique_vals - data[i, feature]).argmin()
                new_ranks[i] = avg_ranks[closest_idx]

            # Step 2: Normalize the ranks to (0, 1) and apply the inverse CDF
            rank_normalized = new_ranks / (self.n_samples + 1)
            transformed_data[:, feature] = torch.tensor(norm.ppf(rank_normalized), dtype=torch.float32)

        return transformed_data

    def inverse_transform(self, transformed_data):
        """Invert the normal scale transformation to recover the original data."""
        n_samples, n_features = transformed_data.shape
        inverse_data = torch.zeros_like(transformed_data)

        for feature in range(n_features):
            # Get unique values and corresponding average ranks
            unique_vals = self.unique_vals_list[feature]
            avg_ranks = self.avg_ranks_list[feature]

            # Sort the transformed data and map back to the original scale
            sorted_transformed_data, sorted_indices = torch.sort(transformed_data[:, feature])

            # Assign sorted transformed data back to original ranks
            rank_indices = torch.argsort(avg_ranks)
            inverse_data[:, feature][sorted_indices] = unique_vals[rank_indices]

        return inverse_data

# Example usage
data = torch.tensor([[3.5, 2.0], [1.2, 5.0], [4.8, 3.0], [2.9, 4.5], [5.0, 1.5]], dtype=torch.float32)
new_data = torch.tensor([[3.6, 2.5], [1.1, 4.8]], dtype=torch.float32)

# Initialize and fit the transformer
transformer = RankInverseNormalTransformer()
transformer.fit(data)

# Transform new data
transformed_new_data = transformer.transform(new_data)

# Optionally invert the transformation for the original data
transformed_data = transformer.transform(data)
recovered_data = transformer.inverse_transform(transformed_data)

print("Original data:\n", data)
print("Transformed original data:\n", transformed_data)
print("Transformed new data:\n", transformed_new_data)
print("Recovered original data:\n", recovered_data)