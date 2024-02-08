import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import numpy as np


class GaussianMixtureTorch:
    """
    Gaussian mixture model for approximating a multivariate joint density function. Uses

    Attributes
    ----------
    data : torch.Tensor
        original data tensor used for fitting the QuantileMVNorm
    qt_data : torch.Tensor
        quantile transformed data
    mvnorm : torch.distributions.MultivariateNormal
        fitted multivariate normal distribution

    Methods
    -------
    fit_multivariate_normal(data)
        Fits a multivariate normal distribution to the data tensor.
    pdf(data)
        Computes the probability density function at the given data points.
    cdf(data)
        Computes the cumulative distribution function at the given data points.
    """

    def __init__(self):
        """
        Initializes a quantile multivariate normal distribution using the provided data tensor.
        """
        self.gmm = None
        self.n_components = None

    def fit(self, data: torch.Tensor, cv_n_components: list[int] = None):
        """
        Fits guassian mixture models using a number of different components to the given data.
        Selects the best number of components using 5 fold cross validation.

        Parameters
        ----------
        data : torch.tensor
            A 2D tensor with shape [num_samples, num_variables]. Each row is a sample,
            each column is a variable.
        cv_n_components : list[int]
            A list with component numbers to be evaluated.

        """
        cv_n_components = [2, 3, 4, 5, 10] if cv_n_components is None else cv_n_components
        if len(cv_n_components) > 1:
            n_folds = 5
            kfold = KFold(n_splits=n_folds)

            numpy_data = data.cpu().numpy()
            # Perform cross-validation for each number of components
            average_log_likelihood = []
            for n in cv_n_components:
                log_likelihoods = []
                for train_index, val_index in kfold.split(numpy_data):
                    # Split the data into training and validation sets
                    data_train, data_val = (
                        numpy_data[train_index],
                        numpy_data[val_index],
                    )

                    # Fit the GMM on the training data
                    gmm = GaussianMixture(n_components=n)
                    gmm.fit(data_train)

                    # Compute the log likelihood on the validation data
                    log_likelihood = gmm.score(data_val)
                    log_likelihoods.append(log_likelihood)

                # Compute the average log likelihood over the k folds
                average_log_likelihood.append(np.mean(log_likelihoods))

            # Select the number of components that maximizes the average log likelihood
            optimal_n_components = cv_n_components[np.argmax(average_log_likelihood)]
            gmm = GaussianMixture(n_components=optimal_n_components)
        else:
            gmm = GaussianMixture(n_components=cv_n_components[0])
        gmm.fit(numpy_data)

        self.n_components = optimal_n_components
        self.gmm = gmm

    def pdf(self, data: torch.tensor):
        """
        Computes the probability density function at the given data points.

        Parameters
        ----------
        data : torch.Tensor
            The tensor of data points at which to evaluate the PDF.

        Returns
        -------
        torch.Tensor
            The PDF evaluated at the given data points.
        """
        if self.gmm is None:
            raise AttributeError(
                "Please fit the model using the fit() method before querying density."
            )

        log_likelihood = self.gmm.score_samples(data.cpu().numpy())

        # Convert log likelihood to actual density
        return torch.from_numpy(np.exp(log_likelihood))
