from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from irtorch.estimation_algorithms import AE, VAE
from irtorch._internal_utils import entropy

if TYPE_CHECKING:
    from irtorch.models.base_irt_model import BaseIRTModel

pio.templates.default = "plotly_white"
logger = logging.getLogger("irtorch")
DEFAULT_COLORSCALE = "Greens"

class Plotter:
    """
    Class for producing various plots from an IRT model.
    A fitted :doc:`model <irt_models>` typically holds an instance of this class in its `plot` property. 
    Thus the methods can be accessed through `model.plot.method_name()`.

    Parameters
    ----------
    model : BaseIRTModel
        The IRT model to use for plotting.
    """
    def __init__(
        self, model: BaseIRTModel
    ):
        self.model = model
        self.linewidth = 2.5
        self.markersize = 9


    def training_history(self) -> go.Figure:
        """
        Plots the training history of the model.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        if all(len(val) == 0 for val in self.model.algorithm.training_history.values()):
            logging.error("Model has not been trained yet")
            raise AttributeError("Model has not been trained yet")

        data_frames = []

        if isinstance(self.model.algorithm, (AE, VAE)):
            x_label = "Iteration interval"
        else:
            x_label = "Iteration"

        if "train_loss" in self.model.algorithm.training_history and len(self.model.algorithm.training_history["train_loss"]) > 0:
            train_df = pd.DataFrame({
                x_label: range(1, len(self.model.algorithm.training_history["train_loss"]) + 1),
                "Loss": self.model.algorithm.training_history["train_loss"],
                "Type": "Training"
            })
            data_frames.append(train_df)

        if "validation_loss" in self.model.algorithm.training_history and len(self.model.algorithm.training_history["validation_loss"]) > 0:
            validation_df = pd.DataFrame({
                x_label: range(1, len(self.model.algorithm.training_history["validation_loss"]) + 1),
                "Loss": self.model.algorithm.training_history["validation_loss"],
                "Type": "Validation"
            })
            data_frames.append(validation_df)

        if not data_frames:
            raise ValueError("No training or validation loss data available for plotting.")

        plot_df = pd.concat(data_frames)

        fig = px.line(
            plot_df, x=x_label, y="Loss", color="Type",
            labels={"Loss": "Loss", x_label: x_label},
            title="Training History")

        if plot_df["Type"].nunique() > 1:
            fig.update_layout(showlegend=True)
        else:
            fig.update_layout(showlegend=False)

        return fig

    @torch.no_grad()
    def latent_score_distribution(
        self,
        scores_to_plot: torch.Tensor = None,
        population_data: torch.Tensor = None,
        latent_variables: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        contour_colorscale: str = DEFAULT_COLORSCALE,
        contour_plot_bins: int = None,
        rescale: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Plots the distribution of latent scores.

        Parameters
        ----------
        scores_to_plot : torch.Tensor, optional
            If provided, the requested latent variable distributions are plotted directly.
            If None, scores are computed from the population data or the model training data. (default is None)
        population_data : torch.Tensor, optional
            The data used to compute the latent scores. If None, uses the training data. (default is None)
        latent_variables : tuple[int], optional
            The latent dimensions to include in the plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Density" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        contour_colorscale : str, optional
            Sets the colorscale for the multiple latent variable contour plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        countor_plot_bins : int, optional
            The number of histogram bins to use for creating the contour plot. (default is None and uses Sturges’ Rule)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`irtorch.models.BaseIRTModel.latent_scores` if scores_to_plot is not provided.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        if len(latent_variables) > 2:
            raise ValueError("Can only plot 1 or 2 latent variables. Select a subset using the latent_variables argument.")
        if len(latent_variables) > self.model.latent_variables:
            raise ValueError(
                f"Cannot plot {len(latent_variables)} dimensions. "
                f"Fitted model is a {self.model.latent_variables} dimensional model."
            )
        if len(latent_variables) > self.model.latent_variables:
            raise ValueError(
                f"Cannot plot latent variable {latent_variables}. "
                f"Fitted model is a {self.model.latent_variables} dimensional model."
            )

        if scores_to_plot is None:
            if population_data is None:
                population_data = self.model.algorithm.train_data
            else:
                population_data = population_data.contiguous()

            scores = self.model.latent_scores(data=population_data, rescale=rescale, **kwargs)
        else:
            scores = scores_to_plot

        if scores[:, [i - 1 for i in latent_variables]].shape[1] > 1:
            x_label = x_label or f"Latent variable {latent_variables[0]}"
            y_label = y_label or f"Latent variable {latent_variables[1]}"

        return self._distribution_plot(
            latent_scores=scores[:, [i - 1 for i in latent_variables]],
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            contour_colorscale=contour_colorscale,
            contour_plot_bins=contour_plot_bins,
        )

    def item_entropy(
        self,
        item: int,
        latent_variables: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = DEFAULT_COLORSCALE,
        theta_range: tuple[float, float] = None,
        second_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None,
        rescale: bool = True,
    ) -> go.Figure:
        """
        Plot the entropy of an item against the latent variable(s).

        Parameters
        ----------
        item : int
            The item for which to plot the entropy.
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None and prints "Item {item} entropy")
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses 'Entropy' for one latent variable, and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        theta_range : tuple[float, float], optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_theta_range : tuple[float, float], optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if theta_range is not None and len(theta_range) != 2:
            raise TypeError("theta_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_theta_range is not None and len(second_theta_range) != 2:
            raise TypeError("second_theta_range needs to have a length of 2 if specified.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [theta - 1 for theta in latent_variables]

        theta_grid = self._get_theta_grid_for_plotting(latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale)
        
        mean_output = self.model(theta_grid)
        item_entropies = entropy(self.model.probabilities_from_output(mean_output))[:, item - 1]

        if rescale and self.model.scale:
            scores_to_plot = self.model.transform_theta(theta_grid)[:, latent_indices]
        else:
            scores_to_plot = theta_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if len(latent_variables) == 1:
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed theta scale
                start_idx = min_indices[0].item()  # get the first index
                scores_to_plot = scores_to_plot[:start_idx]
                item_entropies = item_entropies[:start_idx]
            else:
                start_idx = min_indices[-1].item()  # get the last index
                scores_to_plot = scores_to_plot[start_idx:]
                item_entropies = item_entropies[start_idx:]
                
            fig = self._2d_line_plot(
                x = scores_to_plot,
                y = item_entropies,
                title = title or f"Item {item} entropy",
                x_label = x_label or "Latent variable",
                y_label = y_label or "Entropy",
                color = color or None
            )
            fig.update_yaxes(range=[0, None])
            return fig
        
        if len(latent_variables) == 2:
            grid_size = int(np.sqrt(item_entropies.size()))
            return self._3d_surface_plot(
                x = scores_to_plot[:, 0].reshape((grid_size, grid_size)),
                y = scores_to_plot[:, 1].reshape((grid_size, grid_size)),
                z = item_entropies.reshape((grid_size, grid_size)),
                title = title or f"Item {item} entropy",
                x_label = x_label or "Latent variable 1",
                y_label = y_label or "Latent variable 2",
                z_label = "Entropy",
                colorscale = colorscale
            )

    def log_likelihood(
        self,
        data: torch.Tensor,
        latent_variables: tuple[int] = (1,),
        items: list[int] = None,
        expected_sum_score: bool = False,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = DEFAULT_COLORSCALE,
        theta_range: tuple[float, float] = None,
        second_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None,
        rescale: str = "theta",
    ) -> go.Figure:
        """
        Plots the log-likelihood function against the latent variable(s) for the supplied response pattern.

        Parameters
        ----------
        data : torch.Tensor
            The response data. Needs to be a one row tensor with the same number of columns as the model has items.
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        items : list[int], optional
            The items to consider for computing the log-likelihood. If None, all items in the model are used. (default is None)
        expected_sum_score : bool, optional
            Whether to plot the log-likelihood against the expected sum score instead of the latent variable(s). (default is False)
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Information" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        theta_range : tuple[float, float], optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_theta_range : tuple[float, float], optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis to construct the latent variable grid for which information is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)
        
        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if theta_range is not None and len(theta_range) != 2:
            raise TypeError("theta_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_theta_range is not None and len(second_theta_range) != 2:
            raise TypeError("second_theta_range needs to have a length of 2 if specified.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [theta - 1 for theta in latent_variables]

        theta_grid = self._get_theta_grid_for_plotting(latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale)
        
        if expected_sum_score:
            scores_to_plot = self.model.expected_scores(theta_grid, return_item_scores=False).unsqueeze(1)
        elif rescale and self.model.scale:
            scores_to_plot = self.model.transform_theta(theta_grid)
            scores_to_plot = scores_to_plot[:, latent_indices]
        else:
            scores_to_plot = theta_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        duplicated_data = data.repeat(theta_grid.shape[0], 1)
        log_likelihood = self.model.evaluate.log_likelihood(duplicated_data, theta_grid, reduction="none")
        if items is not None:
            item_mask = torch.zeros(self.model.items, dtype=bool)
            item_mask[[item - 1 for item in items]] = 1
            log_likelihood = log_likelihood[:, item_mask]

        log_likelihood = log_likelihood.sum(dim=1)

        if len(latent_variables) == 1:
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed theta scale
                start_idx = min_indices[0].item()  # get the first index
                scores_to_plot = scores_to_plot[:start_idx]
                log_likelihood = log_likelihood.detach_().squeeze_()[:start_idx]
            else:
                start_idx = min_indices[-1].item()  # get the last index
                scores_to_plot = scores_to_plot[start_idx:]
                log_likelihood = log_likelihood.detach_().squeeze_()[start_idx:]
                
            if x_label is None and expected_sum_score:
                x_label = "Expected sum score"
                
            fig = self._2d_line_plot(
                x = scores_to_plot,
                y = log_likelihood,
                title = title or None,
                x_label = x_label or "Latent variable",
                y_label = y_label or "Log-likelihood",
                color = color or None
            )
            if expected_sum_score:
                fig.add_vline(
                    x=data.sum().item(),
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.7
                )
                fig.add_annotation(
                    x=data.sum().item(),
                    y=1.0,
                    yref="paper",
                    text=f"Observed sum score: {data.sum().item()}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            return fig
        if len(latent_variables) == 2:
            grid_size = int(np.sqrt(log_likelihood.size()))
            return self._3d_surface_plot(
                x = scores_to_plot[:, 0].reshape((grid_size, grid_size)),
                y = scores_to_plot[:, 1].reshape((grid_size, grid_size)),
                z = log_likelihood.reshape((grid_size, grid_size)),
                title = title or "Log-likelihood",
                x_label = x_label or "Latent variable 1",
                y_label = y_label or "Latent variable 2",
                z_label = "Log-likelihood",
                colorscale = colorscale
            )
        
    @torch.no_grad()
    def item_latent_variable_relationships(
        self,
        relationships: torch.Tensor = None,
        theta: torch.Tensor = None,
        title: str = "Relationships: Items vs. latent variables",
        x_label: str = "Latent variable",
        y_label: str = "Items",
        colorscale: str = DEFAULT_COLORSCALE,
    ) -> go.Figure:
        """
        Create a heatmap of item-latent variable relationships. Uses :meth:`irtorch.models.BaseIRTModel.expected_item_score_gradients` to compute the relationships if not provided.

        Parameters
        ----------
        relationships : torch.Tensor
            A tensor of item-latent variable relationships. Each row represents an item and each column represents a latent variable.
            If not provided, the relationships are computed using :meth:`irtorch.models.BaseIRTModel.expected_item_score_gradients`.
        theta : torch.Tensor, optional
            The theta scores to use for computing the relationships. Need to be on the original theta scale.
            If not provided, the training theta scores are used. (default is None)	
        title : str, optional
            The title for the plot. (default is "Relationships: Items vs. latent variables")
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")
        y_label : str, optional
            The label for the Y-axis. (default is "Items")
        colorscale : str, optional
            Sets the colorscale figure. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        if relationships is None:
            if theta is None:
                if self.model.algorithm.training_theta_scores is not None:
                    theta = self.model.algorithm.training_theta_scores
                else:
                    raise ValueError("relationships or theta need to be provided if there are no training theta scores.")
            relationships = self.model.expected_item_score_gradients(theta).mean(dim=0)

        relationships = relationships.numpy()
        
        df = pd.DataFrame(relationships)
        df.columns = [f"{i+1}" for i in range(df.shape[1])]
        df.index = [f"Item {i+1}" for i in range(df.shape[0])]
        
        fig = px.imshow(
            df,
            labels=dict(x=x_label, y=y_label, color="Relationship"),
            x=df.columns,
            y=df.index,
            aspect="auto",
            title=title,
            color_continuous_scale=colorscale
        )

        base_height = 200 # high based on the number of items
        per_item_height = 20
        total_height = base_height + (per_item_height * 80)
        fig.update_layout(height=total_height, width=800)
        
        return fig

    @torch.no_grad()
    def item_probabilities(
        self,
        item: int,
        latent_variables: tuple = (1, ),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        theta_range: tuple[float, float] = None,
        second_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None,
        plot_group_fit: bool = False,
        group_fit_groups: int = 10,
        group_fit_data: int = None,
        group_fit_population_theta: torch.Tensor = None,
        theta_estimation: str = "ML",
        grayscale: bool = False,
        plot_derivative: bool = False,
        rescale: bool = True,
    ) -> go.Figure:
        """
        Plots the item probability curves for a given item. Supports 2D and 3D plots.

        Parameters
        ----------
        item : int
            The item to plot (starts from 1).
        latent_variables : tuple, optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None and uses "IRF - Item {item}")
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Probability")
        theta_range : tuple, optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_theta_range : tuple, optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis used for probability evaluation. (default is None and uses 200 for one latent variable and 25 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        plot_group_fit : bool, optional
            Plot group average probabilities to assess fit. (default is False)
        group_fit_groups : int, optional
            Only for plot_group_fit = True. The number of groups. (default is 10)
        group_fit_data: torch.tensor, optional
            Only for plot_group_fit = True. The data used for group fit plots. Uses training data if not provided. (default is None)
        group_fit_population_theta : torch.tensor, optional
            Only for plot_group_fit = True. The theta scores corresponding to group_fit_data. Will be estimated using group_theta_estimation if not provided. (default is None)
        theta_estimation : str, optional
            Only for plot_group_fit = True. The estimation method for theta. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        grayscale : bool, optional
            Plot the item probability curves in grey scale. (default is False)
        plot_derivative : bool, optional
            Plot the first derivative of the item probability curves. Only for plots with one latent variable. (default is False)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if theta_range is not None and len(theta_range) != 2:
            raise TypeError("theta_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_theta_range is not None and len(second_theta_range) != 2:
            raise TypeError("second_theta_range needs to have a length of 2 if specified.")
        if plot_group_fit and model_dim > 1:
            raise TypeError("Group fit plots are only supported for unidimensional models with one latent variable")
        if steps is None:
            steps = 200 if len(latent_variables) == 1 else 25

        latent_indices = [theta - 1 for theta in latent_variables]

        mask = torch.ones(model_dim, dtype=bool)
        mask[latent_indices] = 0
        if fixed_thetas is None:
            if hasattr(self.model.algorithm, "training_theta_scores") and self.model.algorithm.training_theta_scores is not None:
                fixed_thetas = self.model.algorithm.training_theta_scores[:, mask].median(dim=0).values
            else:
                fixed_thetas = torch.zeros(model_dim)[mask]

        elif len(fixed_thetas) is not model_dim - len(latent_variables):
            raise TypeError("If specified, the number of fixed latent variables needs to be the same as the number of variables in the model not used for plotting.")

        theta_grid = self._get_theta_grid_for_plotting(latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale)

        if rescale and self.model.scale:
            scores_to_plot = self.model.transform_theta(theta_grid)
        else:
            scores_to_plot = theta_grid
        
        if plot_derivative and len(latent_variables) == 1:
            prob_matrix = self.model.probability_gradients(theta_grid, rescale)[:, item - 1, :self.model.item_categories[item - 1], latent_variables[0] - 1]
        else:
            prob_matrix = self.model.item_probabilities(theta_grid)[:, item - 1, :self.model.item_categories[item - 1]]

        if len(latent_variables) == 1:
            if plot_group_fit:
                (
                    group_probs_data,
                    group_probs_model,
                    latent_group_means,
                ) = self.model.evaluate.latent_group_probabilities(
                    data=group_fit_data,
                    theta=group_fit_population_theta,
                    rescale=rescale,
                    latent_variable=latent_variables[0],
                    groups=group_fit_groups,
                    theta_estimation=theta_estimation,
                )

                group_probs_data = group_probs_data[:, item - 1, 0:self.model.item_categories[item - 1]]
                group_probs_model = group_probs_model[:, item - 1, 0:self.model.item_categories[item - 1]]
                
            else:
                group_probs_data = group_probs_model = latent_group_means = None

            return self._item_probabilities_plot(
                scores_to_plot[:, latent_indices[0]],
                prob_matrix,
                latent_group_means,
                group_probs_data,
                group_probs_model,
                title=title or f"IRF - Item {item}",
                x_label=x_label or "Latent variable",
                y_label=y_label or "Probability",
                grayscale=grayscale
            )
        
        if len(latent_variables) == 2:
            return self._item_probabilities_3dplot(
                scores_to_plot[:, latent_indices[0]],
                scores_to_plot[:, latent_indices[1]],
                prob_matrix,
                title=title or f"IRF - Item {item}",
                x_label=x_label or f"Latent variable {latent_variables[0]}",
                y_label=y_label or f"Latent variable {latent_variables[1]}",
                z_label="Probability",
                grayscale=grayscale
            )

    @torch.no_grad()
    def information(
        self,
        items: list[int] = None,
        latent_variables: tuple[int] = (1,),
        degrees: list[int] = None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = DEFAULT_COLORSCALE,
        theta_range: tuple[float, float] = None,
        second_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None,
        rescale: bool = True,
    ) -> go.Figure:
        """
        Plots the Fisher information function against the latent variable(s).
        Supports both item and test information.

        Parameters
        ----------
        items : list[int], optional
            The items to plot. If None, the full test information is plotted. (default is None)
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        degrees : list[int], optional
            A list of angles in degrees between 0 and 90. One degree for each latent variable.
            Only applicable when the model is multidimensional.
            Information will be computed in the direction of the angles. (default is None)
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Information" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        theta_range : tuple[float, float], optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_theta_range : tuple[float, float], optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis to construct the latent variable grid for which information is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if theta_range is not None and len(theta_range) != 2:
            raise TypeError("theta_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_theta_range is not None and len(second_theta_range) != 2:
            raise TypeError("second_theta_range needs to have a length of 2 if specified.")
        if degrees is None and model_dim > 1:
            raise ValueError("Degrees must be provided for multidimensional models.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [theta - 1 for theta in latent_variables]

        theta_grid = self._get_theta_grid_for_plotting(latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale)
        
        if theta_grid.shape[0] > 2000:
            logger.warning("A large grid of latent variable values is used for plotting. This may take a while. Consider lowering the steps argument.")

        if rescale and self.model.scale:
            scores_to_plot = self.model.transform_theta(theta_grid)
        else:
            scores_to_plot = theta_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if items is not None:
            item_mask = torch.zeros(self.model.items, dtype=bool)
            item_mask[[item - 1 for item in items]] = 1
            information = self.model.information(theta_grid, item=True, degrees=degrees, rescale=rescale)[:, item_mask].sum(dim=1)
        else:
            information = self.model.information(theta_grid, item=False, degrees=degrees, rescale=rescale)

        if len(latent_variables) == 1:
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed theta scale
                start_idx = min_indices[0].item()  # get the first index
                scores_to_plot = scores_to_plot[:start_idx]
                information = information.detach_().squeeze_()[:start_idx]
            else:
                start_idx = min_indices[-1].item()  # get the last index
                scores_to_plot = scores_to_plot[start_idx:]
                information = information.detach_().squeeze_()[start_idx:]
                
            return self._2d_line_plot(
                x = scores_to_plot,
                y = information,
                title = title or "Information",
                x_label = x_label or "Latent variable",
                y_label = y_label or "Information",
                color = color or None
            )
        if len(latent_variables) == 2:
            grid_size = int(np.sqrt(information.size()))
            return self._3d_surface_plot(
                x = scores_to_plot[:, 0].reshape((grid_size, grid_size)),
                y = scores_to_plot[:, 1].reshape((grid_size, grid_size)),
                z = information.reshape((grid_size, grid_size)),
                title = title or "Information",
                x_label = x_label or "Latent variable 1",
                y_label = y_label or "Latent variable 2",
                z_label = "Information",
                colorscale = colorscale
            )

    def expected_sum_score(
        self,
        items: list[int] = None,
        latent_variables: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = DEFAULT_COLORSCALE,
        theta_range: tuple[float, float] = None,
        second_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None,
        rescale: str = True,
    ) -> go.Figure:
        """
        Plots the expected sum score from the model against the latent variable(s).
        Supports full test scores, a single item or a subset of items.

        Parameters
        ----------
        items : list[int], optional
            The items used to compte the sum score. If None, all items are used. (default is None)
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Expected sum score" or "Expected item score" for one latent variable, and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        theta_range : tuple[float, float], optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_theta_range : tuple[float, float], optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        rescale : bool, optional
            Whether to plot the transformed latent scores if a transformation scale exists. (default is True)

        Returns
        -------
        go.Figure
            The plotly Figure object for the plot.
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if theta_range is not None and len(theta_range) != 2:
            raise TypeError("theta_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_theta_range is not None and len(second_theta_range) != 2:
            raise TypeError("second_theta_range needs to have a length of 2 if specified.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [theta - 1 for theta in latent_variables]

        theta_grid = self._get_theta_grid_for_plotting(latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale)
        
        if items is not None:
            item_mask = torch.zeros(self.model.items, dtype=bool)
            item_mask[[item - 1 for item in items]] = 1
            sum_scores = self.model.expected_scores(theta_grid, return_item_scores=True)[:, [item - 1 for item in items]].sum(dim=1)
        else:
            sum_scores = self.model.expected_scores(theta_grid, return_item_scores=False)

                
        if rescale and self.model.scale:
            scores_to_plot = self.model.transform_theta(theta_grid)[:, latent_indices]
        else:
            scores_to_plot = theta_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if items is not None and len(items) == 1:
            title = f"Expected score. Item {items[0]}" if title is None else title

        if len(latent_variables) == 1:
            if items is not None and len(items) == 1:
                y_label = "Expected item score" if y_label is None else y_label
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed theta scale
                start_idx = min_indices[0].item()  # get the first index
                scores_to_plot = scores_to_plot[:start_idx]
                sum_scores = sum_scores[:start_idx]
            else:
                start_idx = min_indices[-1].item()  # get the last index
                scores_to_plot = scores_to_plot[start_idx:]
                sum_scores = sum_scores[start_idx:]
                
            fig = self._2d_line_plot(
                x = scores_to_plot,
                y = sum_scores,
                title = title or "Expected sum score",
                x_label = x_label or "Latent variable",
                y_label = y_label or "Expected sum score",
                color = color or None
            )
            fig.update_yaxes(range=[0, None])
            return fig
        
        if len(latent_variables) == 2:
            grid_size = int(np.sqrt(sum_scores.size()))
            return self._3d_surface_plot(
                x = scores_to_plot[:, 0].reshape((grid_size, grid_size)),
                y = scores_to_plot[:, 1].reshape((grid_size, grid_size)),
                z = sum_scores.reshape((grid_size, grid_size)),
                title = title or "Expected sum score",
                x_label = x_label or "Latent variable 1",
                y_label = y_label or "Latent variable 2",
                z_label = "Expected sum score",
                colorscale = colorscale
            )

    def _get_theta_grid_for_plotting(self, latent_variables, theta_range, second_theta_range, steps, fixed_thetas, latent_indices, rescale):
        mask = torch.ones(self.model.latent_variables, dtype=bool)
        mask[latent_indices] = False
        invertible = bool(len(self.model.scale)) and all(scale.invertible for scale in self.model.scale)
        has_training_theta_scores = (
            self.model is not None
            and hasattr(self.model.algorithm, "training_theta_scores")
            and self.model.algorithm.training_theta_scores is not None
        )
        use_transformed_train_theta = invertible and rescale and has_training_theta_scores

        if has_training_theta_scores:
            if use_transformed_train_theta:
                theta_source = self.model.transform_theta(
                    self.model.algorithm.training_theta_scores
                )
            else:
                theta_source = self.model.algorithm.training_theta_scores
        else:
            theta_source = None

        if fixed_thetas is None:    
            if theta_source is not None:
                fixed_thetas = theta_source[:, mask].median(dim=0).values
            else:
                fixed_thetas = torch.zeros(self.model.latent_variables)[mask]

        if theta_range is None:
            if theta_source is not None:
                theta_range = (
                    theta_source[:, latent_variables[0] - 1].min().item(),
                    theta_source[:, latent_variables[0] - 1].max().item(),
                )
            else:
                theta_range = (-3, 3)

        if second_theta_range is None and len(latent_indices) > 1:
            if theta_source is not None:
                second_theta_range = (
                    theta_source[:, latent_variables[1] - 1].min().item(),
                    theta_source[:, latent_variables[1] - 1].max().item(),
                )
            else:
                second_theta_range = (-3, 3)

        latent_theta_1 = torch.linspace(theta_range[0], theta_range[1], steps=steps)
        if len(latent_indices) == 1:
            theta_grid = latent_theta_1.unsqueeze(1).repeat(1, self.model.latent_variables)
            theta_grid[:, mask] = fixed_thetas
        else:
            latent_theta_2 = torch.linspace(
                second_theta_range[0], second_theta_range[1], steps=steps
            )
            latent_theta_1, latent_theta_2 = torch.meshgrid(
                latent_theta_1, latent_theta_2, indexing="ij"
            )
            theta_grid = torch.zeros(latent_theta_1.numel(), self.model.latent_variables)
            theta_grid[:, latent_indices[0]] = latent_theta_1.flatten()
            theta_grid[:, latent_indices[1]] = latent_theta_2.flatten()
            theta_grid[:, mask] = fixed_thetas

        if use_transformed_train_theta:
            theta_grid = self.model.inverse_transform_theta(theta_grid)
        return theta_grid

    def _2d_line_plot(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        title: str,
        x_label: str,
        y_label: str,
        color: str
    ) -> go.Figure:
        df = pd.DataFrame({
            "x": x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy(), 
            "y": y.cpu().detach().numpy() if y.is_cuda else y.detach().numpy()
        })
        fig = px.line(
            df, x="x", y="y", title=title, color_discrete_sequence=[color],
            line_shape='spline'
        )
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        return fig

    def _3d_surface_plot(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        title: str,
        x_label: str,
        y_label: str,
        z_label: str,
        colorscale: str
    ) -> go.Figure:
        x = x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy()
        y = y.cpu().detach().numpy() if y.is_cuda else y.detach().numpy()
        z = z.cpu().detach().numpy() if z.is_cuda else z.detach().numpy()
        fig = go.Figure(data=[
            go.Surface(z=z, x=x, y=y, colorscale=colorscale)
        ])
        fig.update_layout(
            title=title,
            scene = {
                "xaxis": {"title": x_label},
                "yaxis": {"title": y_label},
                "zaxis": {"title": z_label}
            }
        )

        return fig

    def _item_probabilities_plot(
        self,
        latent_scores: torch.Tensor,
        prob_matrix: torch.Tensor,
        latent_group_means: torch.Tensor = None,
        group_probs_data: torch.Tensor = None,
        group_probs_model: torch.Tensor = None,
        x_label: str = "Latent variable",
        y_label: str = "Probability",
        title: str = "IRF",
        grayscale: bool = False,
    ) -> go.Figure:
        """
        Creates a plot of item probabilities against latent variable.

        Parameters
        ----------
        latent_scores : torch.Tensor
            A tensor of latent scores. These form the X-axis values for the plot.
        prob_matrix : torch.Tensor
            A 2D tensor where each row represents a different latent score and each column represents a different item response category. The values in the matrix are the probabilities of each response category at each latent score. These form the Y-axis values for the plot.
        latent_group_means : torch.Tensor, optional
            A tensor of group means. (default is None)
        group_probs_data : torch.Tensor
            A 2D tensor where each row represents a latent score group and each column represents a different item response category. The values in the matrix are the probabilities of each response category at each latent score group retrieved from the data. These form the Y-axis values for the plot.
        group_probs_model : torch.Tensor
            A 2D tensor where each row represents a latent score group and each column represents a different item response category. The values in the matrix are the probabilities of each response category at each latent score group retrieved from the data. These form the Y-axis values for the plot.
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")
        y_label : str, optional
            The label for the Y-axis. (default is "Probability")
        title : str, optional
            The title for the plot. (default is "IRF")
        grayscale : bool, optional
            Grayscale plot. (default is False)

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        min_indices = (latent_scores == latent_scores.min()).nonzero().flatten()
        if min_indices[-1] == len(latent_scores) - 1:  # if we have reversed theta scale
            start_idx = min_indices[0].item()  # get the first index
            latent_scores = latent_scores[:start_idx]
            prob_matrix = prob_matrix[:start_idx, :]
        else:
            start_idx = min_indices[-1].item()  # get the last index
            latent_scores = latent_scores[start_idx:]
            prob_matrix = prob_matrix[start_idx:, :]

        latent_scores = latent_scores.cpu().numpy()
        prob_matrix = prob_matrix.cpu().numpy()
        if latent_group_means is not None:
            latent_group_means = latent_group_means.cpu().numpy()
        if group_probs_data is not None:
            group_probs_data = group_probs_data.cpu().numpy()
        if group_probs_model is not None:
            group_probs_model = group_probs_model.cpu().numpy()

        fig = go.Figure()
        num_categories = prob_matrix.shape[1]

        if grayscale:
            colors = self._generate_grayscale_colors(num_categories)
        else:
            colors = px.colors.qualitative.Plotly
        
        # Adjust the size of the palette if there are more categories than colors
        if len(colors) < num_categories:
            colors = colors * (num_categories // len(colors) + 1)

        # Plot each response category
        for i in range(prob_matrix.shape[1]):
            response_text = f"Option {i+1}" if self.model.mc_correct is not None else f"{i}"

            color = colors[i % len(colors)]  # Ensure color wraps around if more categories than colors
            fig.add_trace(go.Scatter(
                x=latent_scores,
                y=prob_matrix[:, i],
                mode="lines",
                name=response_text,
                line=dict(color=color)
            ))

            if latent_group_means is not None and group_probs_data is not None and group_probs_model is not None:
                # Adding scatter plot for group data
                fig.add_trace(go.Scatter(
                    x=latent_group_means,
                    y=group_probs_data[:, i],
                    mode="markers", name="Data",
                    marker=dict(symbol="circle-open", color=color)
                ))
                # Adding scatter plot for group model predictions
                fig.add_trace(go.Scatter(
                    x=latent_group_means,
                    y=group_probs_model[:, i],
                    mode="markers",
                    name="Model",
                    marker=dict(symbol="circle", color=color)
                ))

        legend_title = "Item response" if self.model.mc_correct is not None else "Score"
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=legend_title,
        )

        return fig

    def _item_probabilities_3dplot(
        self,
        latent_variable_x: torch.Tensor,
        latent_variable_y: torch.Tensor,
        prob_matrix: torch.Tensor,
        title: str = "IRF",
        x_label: str = "Latent variable",
        y_label: str = "Latent variable",
        z_label: str = "Probability",
        grayscale: bool = False,
    ) -> go.Figure:
        """
        Creates a 3D plot of item probabilities against two latent variables.

        Parameters
        ----------
        latent_variable_x : torch.Tensor
            A tensor of latent scores for the X-axis.
        latent_variable_y : torch.Tensor
            A tensor of latent scores for the Y-axis.
        prob_matrix : torch.Tensor
            A 2D tensor with the probabilities. These form the Z-axis values for the plot.
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")
        y_label : str, optional
            The label for the Y-axis. (default is "Latent variable")
        z_label : str, optional
            The label for the Z-axis. (default is "Probability")
        title : str, optional
            The title for the plot. (default is "IRF")
        grayscale : bool, optional
            Grayscale plot. (default is False)

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        latent_variable_x = latent_variable_x.cpu().numpy()
        latent_variable_y = latent_variable_y.cpu().numpy()
        prob_matrix = prob_matrix.cpu().numpy()
        # The number of steps (points in each dimension) should be the square root of the number of points.
        steps = int(np.sqrt(len(latent_variable_x)))
        num_responses = prob_matrix.shape[1]

        # Reshape the data into a format suitable for a surface plot
        x = np.array(latent_variable_x).reshape((steps, steps))
        y = np.array(latent_variable_y).reshape((steps, steps))

        if grayscale:
            colors = self._generate_grayscale_colors(num_responses)
        else:
            colors = px.colors.qualitative.Plotly

        # Ensure we have enough colors for the number of responses
        if len(colors) < num_responses:
            colors += colors * (num_responses // len(colors) + 1) 

        fig = go.Figure()
        for response_category in range(num_responses):
            z = prob_matrix[:, response_category].reshape((steps, steps))
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                name=f"Response {response_category}",
                showscale=False,  # Optionally hide the color scale
                colorscale=[(0, colors[response_category % len(colors)]), (1, colors[response_category % len(colors)])],
            ))
            # Dummy trace for legend
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=10, color=colors[response_category % len(colors)]),
                showlegend=True, name=f"Response {response_category}"
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
            ),
            legend_title="Item responses",
            autosize=True,
        )

        return fig

    def _distribution_plot(
        self,
        latent_scores: torch.Tensor,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        contour_colorscale: str = DEFAULT_COLORSCALE,
        contour_plot_bins = None,
    ) -> go.Figure:
        """
        Plots the latent score distribution.

        Parameters
        ----------
        latent_scores : torch.Tensor
            A tensor of latent scores for which to plot the kernel density estimate.
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            For more than one latent variable. The label for the Y-axis. (default is None and uses "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        contour_colorscale : str, optional
            Sets the colorscale for the multiple latent variable contour plots. See https://plotly.com/python/builtin-colorscales/ (default is "Greens")
        countor_plot_bins : int, optional
            The number of histogram bins to use for creating the contour plot. (default is None and uses Sturges’ Rule)
        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        latent_scores = latent_scores.cpu().numpy()
        latent_variables = latent_scores.shape[1]

        if latent_variables > 2:
            raise ValueError("Can not plot for more than two latent score dimensions.")

        if latent_variables == 1:
            if x_label is None:
                x_label = "Latent variable"
            return self._single_latent_variable_distribution_plot(latent_scores.squeeze(), title, x_label, color)
        else:
            if x_label is None:
                x_label = "Latent variable 1"
            if y_label is None:
                y_label = "Latent variable 2"
            return self._two_latent_variables_distribution_plot(latent_scores, title, x_label, y_label, contour_colorscale, contour_plot_bins)

    def _single_latent_variable_distribution_plot(
        self,
        scores: np.ndarray,
        title: str,
        x_label: str,
        color: str
    ) -> go.Figure:
        df = pd.DataFrame(scores, columns=["values"])
        histogram_kwargs = {
            "x": "values",
            "marginal": "box"
        }
        if color:
            histogram_kwargs["color_discrete_sequence"] = [color]
        fig = px.histogram(df, **histogram_kwargs)
        fig.update_layout(title=title, xaxis_title=x_label)
        return fig

    def _two_latent_variables_distribution_plot(
        self,
        scores: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        contour_colorscale: str,
        contour_plot_bins: int
    ) -> go.Figure:
        if contour_plot_bins is None:
            contour_plot_bins = int(np.log2(scores.shape[0])) + 1 # Sturges’ Rule
        histogram2d, x_edges, y_edges = np.histogram2d(scores[:, 0], scores[:, 1], bins=contour_plot_bins, density=True)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        fig = go.Figure(data =
            go.Contour(
                z=histogram2d.T,
                x=x_centers, # Centers of bins (x-axis)
                y=y_centers, # Centers of bins (y-axis)
                colorscale=contour_colorscale
            )
        )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig

    def _generate_grayscale_colors(self, n, start_color="#b0b0b0"):
        """
        Generates a list of n colors in grayscale from light gray to black.

        Parameters
        ----------
        n : int
            The number of colors to generate.
        start_color : str, optional
            The color to start from. (default is '#b0b0b0')

        Returns
        -------
        list
            A list of n colors in hexadecimal format.
        """
        start_rgb = np.array([int(start_color[i:i+2], 16) for i in (1, 3, 5)])  # Convert start_color to RGB
        end_rgb = np.array([0, 0, 0])  # RGB for black
        colors = [(
            start_rgb + (end_rgb - start_rgb) * i / (n - 1)
        ).astype(int) for i in range(n)]
        # Convert RGB to hexadecimal
        hex_colors = ["#" + "".join(f"{int(c):02x}" for c in color) for color in colors]
        return hex_colors

    @torch.no_grad()
    def scale_transformations(
        self,
        input_latent_variable: int = 1,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = DEFAULT_COLORSCALE,
        input_theta_range: tuple[float, float] = None,
        steps: int = None,
        fixed_thetas: torch.Tensor = None
    ) -> go.Figure:
        """
        Plots the scale transformations for the latent variable(s).

        Parameters
        ----------
        input_latent_variable : int, optional
            The latent variable to visualize (the input dimension). For multidimensional models, other latent variables are fixed (see fixed_thetas below). (default is 1)
        title : str, optional
            The title for the plot. (default is None and uses "Scale Transformation(s)")
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Original scale")
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Transformed scale")
        input_theta_range : tuple[float, float], optional
            The theta range for plotting. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        second_input_theta_range : tuple[float, float], optional
            The range for plotting for the second latent variable. For invertible scale transformations, this is the range of the transformed theta scores. Otherwise it is the range of the original theta scores. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each theta axis to construct the latent variable grid for which information is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_thetas: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        model_dim = self.model.latent_variables
        if input_latent_variable > model_dim:
            raise TypeError(f"Cannot plot latent variable {input_latent_variable} with a {model_dim}-dimensional model.")
        if input_theta_range is not None and len(input_theta_range) != 2:
            raise TypeError("input_theta_range needs to have a length of 2.")
        if steps is None:
            steps = 100

        theta_grid = self._get_theta_grid_for_plotting(
            (input_latent_variable,),
            input_theta_range,
            None,
            steps,
            fixed_thetas,
            [input_latent_variable-1],
            rescale=False
        )

        if self.model.scale:
            transformed_theta = self.model.transform_theta(theta_grid)
        else:
            raise ValueError("No scale transformations available.")

        original_vals = theta_grid[:, input_latent_variable-1]
        fig = go.Figure()
        default_colors = px.colors.qualitative.Plotly
        for i in range(model_dim):
            trace_color = color if color is not None else default_colors[i % len(default_colors)]
            fig.add_trace(go.Scatter(
                x=original_vals.cpu().detach().numpy(),
                y=transformed_theta[:, i].cpu().detach().numpy(),
                mode="lines",
                name=f"Latent variable {i+1}",
                line=dict(color=trace_color)
            ))
        fig.update_layout(
            title=title or "Scale Transformation(s)",
            xaxis_title=x_label or "Original scale",
            yaxis_title=y_label or "Transformed scale",
            legend_title="Transformed latent variable"
        )
        return fig
