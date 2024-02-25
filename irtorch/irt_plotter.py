import logging
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.irt_scorer import IRTScorer
from irtorch.irt_evaluator import IRTEvaluator
from irtorch.helper_functions import output_to_item_entropy

pio.templates.default = "plotly_white"
logger = logging.getLogger('irtorch')

class IRTPlotter:
    """
    Initializes the IRTPlotter class with a given model, fitting algorithm, scorer and evaluator.

    Parameters
    ----------
    model : BaseIRTModel
        BaseIRTModel object.
    algorithm : BaseIRTAlgorithm
        BaseIRTAlgorithm object.
    scorer : IRTScorer
        IRTScorer object used to obtain latent variable scores.
    evaluator : IRTEvaluator
        IRTEvaluator object used to obtain evaluation measures.
    """
    def __init__(
        self, model: BaseIRTModel, algorithm: BaseIRTAlgorithm, scorer: IRTScorer, evaluator: IRTEvaluator
    ):
        self.model = model
        self.algorithm = algorithm
        self.scorer = scorer
        self.evaluator = evaluator
        self.linewidth = 2.5
        self.markersize = 9
        self.color_map = "tab10"


    def plot_training_history(self) -> go.Figure:
        """
        Plots the training history of the model.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        if all(len(val) == 0 for val in self.algorithm.training_history.values()):
            logging.error("Model has not been trained yet")
            raise AttributeError("Model has not been trained yet")

        data_frames = []

        if 'train_loss' in self.algorithm.training_history and len(self.algorithm.training_history['train_loss']) > 0:
            train_df = pd.DataFrame({
                'Epoch': range(1, len(self.algorithm.training_history['train_loss']) + 1),
                'Loss': self.algorithm.training_history['train_loss'],
                'Type': 'Training'
            })
            data_frames.append(train_df)

        if 'validation_loss' in self.algorithm.training_history and len(self.algorithm.training_history['validation_loss']) > 0:
            validation_df = pd.DataFrame({
                'Epoch': range(1, len(self.algorithm.training_history['validation_loss']) + 1),
                'Loss': self.algorithm.training_history['validation_loss'],
                'Type': 'Validation'
            })
            data_frames.append(validation_df)

        if not data_frames:
            raise ValueError("No training or validation loss data available for plotting.")

        plot_df = pd.concat(data_frames)

        fig = px.line(
            plot_df, x='Epoch', y='Loss', color='Type',
            labels={'Loss': 'Loss', 'Epoch': 'Epoch'},
            title='Training History')

        return fig

    @torch.inference_mode()
    def plot_latent_score_distribution(
        self,
        scores_to_plot: torch.Tensor = None,
        population_data: torch.Tensor = None,
        scale: str = "bit",
        latent_variables_to_plot: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        contour_colorscale: str = "Plasma",
        contour_plot_bins: int = None,
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
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables_to_plot : tuple[int], optional
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
            Sets the colorscale for the multiple latent variable contour plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        countor_plot_bins : int, optional
            The number of histogram bins to use for creating the contour plot. (default is None and uses Sturges’ Rule)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the latent_scores method if scores_to_plot is not provided.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        if len(latent_variables_to_plot) > 2:
            raise ValueError("Can only plot 1 or 2 latent variables. Select a subset using the latent_variables_to_plot argument.")
        if len(latent_variables_to_plot) > self.model.latent_variables:
            raise ValueError(
                f"Cannot plot {len(latent_variables_to_plot)} dimensions. "
                f"Fitted model is a {self.model.latent_variables} dimensional model."
            )
        if len(latent_variables_to_plot) > self.model.latent_variables:
            raise ValueError(
                f"Cannot plot latent variable {latent_variables_to_plot}. "
                f"Fitted model is a {self.model.latent_variables} dimensional model."
            )

        if scores_to_plot is None:
            if population_data is None:
                population_data = self.algorithm.train_data
            else:
                population_data = population_data.contiguous()

            scores = self.scorer.latent_scores(
                data=population_data,
                scale=scale,
                **kwargs
            )
        else:
            scores = scores_to_plot

        return self._distribution_plot(
            latent_scores=scores[:, [i - 1 for i in latent_variables_to_plot]],
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            contour_colorscale=contour_colorscale,
            contour_plot_bins=contour_plot_bins,
        )

    def plot_item_entropy(
        self,
        item: int,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = "Plasma",
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        steps: int = None,
        fixed_zs: torch.Tensor = None,
        **kwargs
    ) -> go.Figure:
        """
        Plot the entropy of an item against the latent variable(s).

        Parameters
        ----------
        item : int
            The item for which to plot the entropy.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses 'Entropy' for one latent variable, and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 

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
        if z_range is not None and len(z_range) != 2:
            raise TypeError("z_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_z_range is not None and len(second_z_range) != 2:
            raise TypeError("second_z_range needs to have a length of 2 if specified.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [z - 1 for z in latent_variables]

        z_grid = self._get_z_grid_for_plotting(latent_variables, z_range, second_z_range, steps, fixed_zs, latent_indices)
        
        mean_output = self.model(z_grid)
        item_entropies = output_to_item_entropy(
            mean_output, self.model.modeled_item_responses
        )[:, item - 1]

        if scale == "bit":
            scores_to_plot = self.scorer.bit_scores_from_z(
                z=z_grid,
                **kwargs
            )[0][:, latent_indices]
        else:
            scores_to_plot = z_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if len(latent_variables) == 1:
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed z scale
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
        
    @torch.inference_mode()
    def plot_item_latent_variable_relationships(
        self,
        relationships: torch.Tensor,
        title: str = "Relationships: Items vs. latent variables",
        x_label: str = "Latent variable",
        y_label: str = "Items",
        colorscale: str = "Plasma",
    ) -> go.Figure:
        """
        Create a heatmap of item-latent variable relationships.

        Parameters
        ----------
        relationships : torch.Tensor
            A tensor of item-latent variable relationships. Typically the returned tensor from expected_item_score_slopes() where each row represents an item and each column represents a latent variable.
        title : str, optional
            The title for the plot. (default is "Relationships: Items vs. latent variables")
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")
        y_label : str, optional
            The label for the Y-axis. (default is "Items")
        colorscale : str, optional
            Sets the colorscale figure. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
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

    @torch.inference_mode()
    def plot_item_probabilities(
        self,
        item: int,
        scale: str = "bit",
        latent_variables: tuple = (1, ),
        fixed_zs: torch.Tensor = None,
        steps: int = 1000,
        bit_score_start_z: torch.tensor = None,
        bit_score_grid_points: int = 300,
        bit_score_z_grid_method: int = "ML",
        bit_score_start_z_guessing_probabilities: list[float] = None,
        bit_score_start_z_guessing_iterations: int = 10000,
        bit_score_items: list[int] = None,
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        plot_group_fit: bool = False,
        group_fit_groups: int = 10,
        group_fit_data: int = None,
        group_fit_population_z: torch.Tensor = None,
        group_z_estimation_method: str = "ML",
        plot_derivative: bool = False,
        grayscale: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the item probability curves for a given item. Supports 2D and 3D plots.

        Parameters
        ----------
        item : int
            The item to plot.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple, optional
            The latent variables to plot. (default is (1,))
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        steps : int, optional
            The number of steps along each z axis used for probability evaluation. (default is 1000)
        bit_score_start_z : int, optional
            The z score used as the starting point for bit score computation. Computed automatically if not provided. (default is 'None')
        bit_score_grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        bit_score_z_grid_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        bit_score_start_z_guessing_probabilities: list[float], optional
            Custom guessing probabilities for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        bit_score_start_z_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)
        bit_score_items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is 'None' and uses all items)
        z_range : tuple, optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple, optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        plot_group_fit : bool, optional
            Plot group average probabilities to assess fit. (default is False)
        group_fit_groups : int, optional
            Only for plot_group_fit = True. The number of groups. (default is 10)
        group_fit_data: torch.tensor, optional
            Only for plot_group_fit = True. The data used for group fit plots. Uses training data if not provided. (default is None)
        group_fit_population_z : torch.tensor, optional
            Only for plot_group_fit = True. The z scores corresponding to group_fit_data. Will be estimated using group_z_estimation_method if not provided. (default is None)
        group_z_estimation_method : str, optional
            The method used for computing z-scores for the groups. Can be 'NN', 'ML', 'MAP' or 'EAP'. (default is 'ML')
        plot_derivative : bool, optional
            If true, plots the derivative of the item probabilitiy curves. Note that this feature is not yet implemented and will raise a TypeError if set to true. (default is False)
        grayscale : bool, optional
            Grayscale plot. (default is False)

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        # TODO: Integrate over non-plotted dimensions for multidimensional models...
        if plot_derivative:
            raise TypeError("Derivatives not yet implemented.")

        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if z_range is not None and len(z_range) != 2:
            raise TypeError("z_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_z_range is not None and len(second_z_range) != 2:
            raise TypeError("second_z_range needs to have a length of 2 if specified.")

        latent_indices = [z - 1 for z in latent_variables]

        mask = torch.ones(model_dim, dtype=bool)
        mask[latent_indices] = 0
        if fixed_zs is None:
            fixed_zs = self.algorithm.training_z_scores[:, mask].median(dim=0).values

        elif len(fixed_zs) is not model_dim - len(latent_variables):
            raise TypeError("If specified, the number of fixed latent variables needs to be the same as the number of variables in the model not used for plotting.")

        min_z, max_z = self.scorer.min_max_z_for_integration()
        if z_range is None:
            z_range = min_z[latent_indices[0]].item(), max_z[latent_indices[0]].item()
        if second_z_range is None and len(latent_indices) > 1:
            second_z_range = min_z[latent_indices[1]].item(), max_z[latent_indices[1]].item()

        latent_z_1 = torch.linspace(z_range[0], z_range[1], steps=steps)
        if len(latent_indices) == 1:
            z_grid = latent_z_1.unsqueeze(1).repeat(1, model_dim)
            z_grid[:, mask] = fixed_zs
        else:
            latent_z_2 = torch.linspace(second_z_range[0], second_z_range[1], steps=steps)
            latent_z_1, latent_z_2 = torch.meshgrid(latent_z_1, latent_z_2, indexing="ij")
            z_grid = torch.zeros(latent_z_1.numel(), model_dim)
            z_grid[:, latent_indices[0]] = latent_z_1.flatten()
            z_grid[:, latent_indices[1]] = latent_z_2.flatten()
            z_grid[:, mask] = fixed_zs
            
        prob_matrix = self.model.item_probabilities(z_grid)[:, item - 1, :self.model.modeled_item_responses[item - 1]]

        if scale == "bit":
            scores_to_plot, bit_score_start_z = self.scorer.bit_scores_from_z(
                z=z_grid,
                start_z=bit_score_start_z,
                one_dimensional=False,
                z_estimation_method=bit_score_z_grid_method,
                # ml_map_device=ml_map_device,
                grid_points=bit_score_grid_points,
                items=bit_score_items,
                start_z_guessing_probabilities=bit_score_start_z_guessing_probabilities,
                start_z_guessing_iterations=bit_score_start_z_guessing_iterations,
            )
        else:
            scores_to_plot = z_grid

        if len(latent_variables) == 1:
            if plot_group_fit:
                (
                    group_probs_data,
                    group_probs_model,
                    latent_group_means,
                ) = self.evaluator.latent_group_probabilities(
                    groups=group_fit_groups,
                    data=group_fit_data,
                    scale=scale,
                    z=group_fit_population_z,
                    z_estimation_method=group_z_estimation_method,
                    bit_score_start_z=bit_score_start_z,
                    bit_score_grid_points=bit_score_grid_points,
                    bit_score_z_grid_method=bit_score_z_grid_method,
                    bit_score_start_z_guessing_probabilities=bit_score_start_z_guessing_probabilities,
                    bit_score_start_z_guessing_iterations=bit_score_start_z_guessing_iterations,
                    bit_score_items=bit_score_items,
                    latent_variable=latent_variables[0],
                )
                group_probs_data = group_probs_data[:, item - 1, 0:self.model.modeled_item_responses[item - 1]]
                group_probs_model = group_probs_model[:, item - 1, 0:self.model.modeled_item_responses[item - 1]]
                
            else:
                group_probs_data = group_probs_model = latent_group_means = None

            return self._item_probabilities_plot(
                scores_to_plot[:, latent_indices[0]],
                prob_matrix,
                latent_group_means,
                group_probs_data,
                group_probs_model,
                title=f"IRF - Item {item}",
                grayscale=grayscale,
            )
        
        if len(latent_variables) == 2:
            return self._item_probabilities_3dplot(
                scores_to_plot[:, latent_indices[0]],
                scores_to_plot[:, latent_indices[1]],
                prob_matrix,
                x_label=f"Latent variable {latent_variables[0]}",
                y_label=f"Latent variable {latent_variables[1]}",
                title=f"IRF - Item {item}",
                grayscale=grayscale,
            )

    def plot_information(
        self,
        items: list[int] = None,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
        degrees: list[int] = None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = "Plasma",
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        steps: int = None,
        fixed_zs: torch.Tensor = None,
        **kwargs
    ) -> go.Figure:
        """
        Plots the Fisher information function against the latent variable(s).
        Supports both item and test information.

        Parameters
        ----------
        items : list[int], optional
            The items to plot. If None, the full test information is plotted. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
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
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which information is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 
        """
        model_dim = self.model.latent_variables
        if len(latent_variables) > 2:
            raise TypeError("Cannot plot more than two latent variables in one plot.")
        if len(latent_variables) > model_dim:
            raise TypeError(f"Cannot plot {len(latent_variables)} latent variables with a {model_dim}-dimensional model.")
        if not all(num <= model_dim for num in latent_variables):
            raise TypeError(f"The latent variables to plot need to be smaller than or equal to {model_dim} (the number of variabels in the model).")
        if z_range is not None and len(z_range) != 2:
            raise TypeError("z_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_z_range is not None and len(second_z_range) != 2:
            raise TypeError("second_z_range needs to have a length of 2 if specified.")
        if degrees is None and model_dim > 1:
            raise ValueError("Degrees must be provided for multidimensional models.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [z - 1 for z in latent_variables]

        z_grid = self._get_z_grid_for_plotting(latent_variables, z_range, second_z_range, steps, fixed_zs, latent_indices)
        
        if z_grid.shape[0] > 2000:
            logger.warning("A large grid of latent variable values is used for plotting. This may take a while. Consider lowering the steps argument.")

        if items is not None:
            item_mask = torch.zeros(self.model.items, dtype=bool)
            item_mask[[item - 1 for item in items]] = 1
            information = self.model.information(z_grid, item=True, degrees=degrees)[:, item_mask].sum(dim=1)
        else:
            information = self.model.information(z_grid, item=False, degrees=degrees)

        if scale == "bit":
            scores_to_plot = self.scorer.bit_scores_from_z(
                z=z_grid,
                **kwargs
            )[0][:, latent_indices]
        else:
            scores_to_plot = z_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if len(latent_variables) == 1:
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed z scale
                start_idx = min_indices[0].item()  # get the first index
                scores_to_plot = scores_to_plot[:start_idx]
                information = information[:start_idx]
            else:
                start_idx = min_indices[-1].item()  # get the last index
                scores_to_plot = scores_to_plot[start_idx:]
                information = information[start_idx:]
                
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

    def plot_expected_sum_score(
        self,
        items: list[int] = None,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = "Plasma",
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        steps: int = None,
        fixed_zs: torch.Tensor = None,
        **kwargs
    ) -> go.Figure:
        """
        Plots the expected sum score from the model against the latent variable(s).
        Supports full test scores, a single item or a subset of items.

        Parameters
        ----------
        items : list[int], optional
            The items used to compte the sum score. If None, all items are used. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
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
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 

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
        if z_range is not None and len(z_range) != 2:
            raise TypeError("z_range needs to have a length of 2.")
        if len(latent_variables) == 1 and second_z_range is not None and len(second_z_range) != 2:
            raise TypeError("second_z_range needs to have a length of 2 if specified.")
        if steps is None:
            steps = 100 if len(latent_variables) == 1 else 18

        latent_indices = [z - 1 for z in latent_variables]

        z_grid = self._get_z_grid_for_plotting(latent_variables, z_range, second_z_range, steps, fixed_zs, latent_indices)
        
        if items is not None:
            item_mask = torch.zeros(self.model.items, dtype=bool)
            item_mask[[item - 1 for item in items]] = 1
            sum_scores = self.model.expected_item_sum_score(z_grid, return_item_scores=True)[:, [item - 1 for item in items]].sum(dim=1)
        else:
            sum_scores = self.model.expected_item_sum_score(z_grid, return_item_scores=False)

        if scale == "bit":
            scores_to_plot = self.scorer.bit_scores_from_z(
                z=z_grid,
                **kwargs
            )[0][:, latent_indices]
        else:
            scores_to_plot = z_grid[:, [var - 1 for var in latent_variables]]
            if scores_to_plot.dim() == 1:
                scores_to_plot = scores_to_plot.unsqueeze(1)

        if items is not None and len(items) == 1:
            title = f"Expected score. Item {items[0]}" if title is None else title

        if len(latent_variables) == 1:
            if items is not None and len(items) == 1:
                y_label = "Expected item score" if y_label is None else y_label
            scores_to_plot.squeeze_()
            min_indices = (scores_to_plot == scores_to_plot.min()).nonzero().flatten()
            if min_indices[-1] == len(scores_to_plot) - 1:  # if we have reversed z scale
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

    def _get_z_grid_for_plotting(self, latent_variables, z_range, second_z_range, steps, fixed_zs, latent_indices):
        mask = torch.ones(self.model.latent_variables, dtype=bool)
        mask[latent_indices] = 0
        if fixed_zs is None:
            fixed_zs = self.algorithm.training_z_scores[:, mask].median(dim=0).values
        
        if z_range is None:
            z_range = (
                self.algorithm.training_z_scores[:, latent_variables[0] - 1].min().item(),
                self.algorithm.training_z_scores[:, latent_variables[0] - 1].max().item()
            )
        if second_z_range is None and len(latent_indices) > 1:
            second_z_range = (
                self.algorithm.training_z_scores[:, latent_variables[1] - 1].min().item(),
                self.algorithm.training_z_scores[:, latent_variables[1] - 1].max().item()
            )

        latent_z_1 = torch.linspace(z_range[0], z_range[1], steps=steps)
        if len(latent_indices) == 1:
            z_grid = latent_z_1.unsqueeze(1).repeat(1, self.model.latent_variables)
            z_grid[:, mask] = fixed_zs
        else:
            latent_z_2 = torch.linspace(second_z_range[0], second_z_range[1], steps=steps)
            latent_z_1, latent_z_2 = torch.meshgrid(latent_z_1, latent_z_2, indexing="ij")
            z_grid = torch.zeros(latent_z_1.numel(), self.model.latent_variables)
            z_grid[:, latent_indices[0]] = latent_z_1.flatten()
            z_grid[:, latent_indices[1]] = latent_z_2.flatten()
            z_grid[:, mask] = fixed_zs
        
        return z_grid

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
            'x': x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy(), 
            'y': y.cpu().detach().numpy() if y.is_cuda else y.detach().numpy()
        })
        fig = px.line(df, x='x', y='y', title=title, color_discrete_sequence=[color])
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
    ) -> tuple[plt.Figure, plt.Axes]:
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
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        min_indices = (latent_scores == latent_scores.min()).nonzero().flatten()
        if min_indices[-1] == len(latent_scores) - 1:  # if we have reversed z scale
            start_idx = min_indices[0].item()  # get the first index
            latent_scores = latent_scores[:start_idx]
            prob_matrix = prob_matrix[:start_idx, :]
        else:
            start_idx = min_indices[-1].item()  # get the last index
            latent_scores = latent_scores[start_idx:]
            prob_matrix = prob_matrix[start_idx:, :]

        plot_group_fit = False
        latent_scores = latent_scores.cpu().numpy()
        prob_matrix = prob_matrix.cpu().numpy()
        if (
            latent_group_means is not None
            and group_probs_data is not None
            and group_probs_model is not None
        ):
            plot_group_fit = True
            latent_group_means = latent_group_means.cpu().numpy()
            group_probs_data = group_probs_data.cpu().numpy()
            group_probs_model = group_probs_model.cpu().numpy()

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlim(min(latent_scores), max(latent_scores))

        if grayscale:
            cmap = plt.get_cmap("binary")
            colors = cmap(torch.linspace(0.3, 1, prob_matrix.shape[1]).numpy())
        else:
            cmap = plt.get_cmap(self.color_map)
            colors = cmap(torch.linspace(0, 1, prob_matrix.shape[1]).numpy())

        # Create empty lists for custom legend handles and labels
        handles = []
        labels = []

        for response_category in range(prob_matrix.shape[1]):
            color = colors[response_category]
            (line,) = ax.plot(
                latent_scores,
                prob_matrix[:, response_category],
                color=color,
                linewidth=self.linewidth,
            )
            if plot_group_fit:
                scatter_data = ax.scatter(
                    latent_group_means,
                    group_probs_data[:, response_category],
                    marker="o",
                    edgecolors=color,
                    facecolors="none",
                    s=self.markersize**2,  # size
                    zorder=3,  # draw dots above grid, default is 2
                )
                scatter_model = ax.scatter(
                    latent_group_means,
                    group_probs_model[:, response_category],
                    marker="o",
                    color=color,
                    s=self.markersize**2,
                    zorder=3,
                )
                handles.append((line, scatter_data, scatter_model))
            else:
                handles.append(line)

            # Add custom legend handle and label
            labels.append(response_category)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if plot_group_fit:
            # Create a custom legend
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=self.markersize,
                    color="black",
                    linestyle="None",
                    markerfacecolor="none",
                    label="Data",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=self.markersize,
                    color="black",
                    linestyle="None",
                    markerfacecolor="black",
                    label="Model",
                ),
            ]
            ax.legend(handles=legend_elements, loc="lower right")

        # Add the line legend
        line_legend = Legend(ax, handles, labels, title="Scores", loc="upper right")
        ax.add_artist(line_legend)

        return fig, ax

    def _item_probabilities_3dplot(
        self,
        latent_variable_x: torch.Tensor,
        latent_variable_y: torch.Tensor,
        prob_matrix: torch.Tensor,
        x_label: str = "Latent variable",
        y_label: str = "Latent variable",
        z_label: str = "Probability",
        title: str = "IRF",
        grayscale: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
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
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        latent_variable_x = latent_variable_x.cpu().numpy()
        latent_variable_y = latent_variable_y.cpu().numpy()
        prob_matrix = prob_matrix.cpu().numpy()
        num_responses = prob_matrix.shape[1]

        # The number of steps (points in each dimension) should be the square root of the number of points.
        steps = int(np.sqrt(len(latent_variable_x)))
        # Reshape the data into a format suitable for a surface plot.
        latent_variable_x = np.array(latent_variable_x).reshape((steps, steps))
        latent_variable_y = np.array(latent_variable_y).reshape((steps, steps))

        if grayscale:
            cmap = plt.get_cmap("binary")
            colors = cmap(torch.linspace(0.3, 1, num_responses).numpy())
        else:
            cmap = plt.get_cmap(self.color_map)
            colors = cmap(torch.linspace(0, 1, num_responses).numpy())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        handles = []
        for response_category in range(num_responses):
            category_probs = prob_matrix[:, response_category].reshape((steps, steps))
            ax.plot_surface(
                latent_variable_x,
                latent_variable_y,
                category_probs,
                color=colors[response_category],
                shade=False,  # Avoids the darkening effect of shading
                alpha=0.7)  # For better visualization of overlaying surfaces

            # Create a proxy artist for the legend entry (since you cannot directly use a 3D surface)
            proxy = Patch(facecolor=colors[response_category], label=response_category)
            handles.append(proxy)
            
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title(title)
        ax.legend(handles=handles, loc="upper right", title="Scores", bbox_to_anchor=(1.15, 1.1))

        return fig, ax

    def _distribution_plot(
        self,
        latent_scores: torch.Tensor,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        contour_colorscale: str = "Plasma",
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
            The label for the Y-axis. (default is None and uses "Density" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        contour_colorscale : str, optional
            Sets the colorscale for the multiple latent variable contour plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
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
            if y_label is None:
                y_label = "Density"
            return self._single_latent_variable_distribution_plot(latent_scores.squeeze(), title, x_label, y_label, color)
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
        y_label: str,
        color: str
    ) -> go.Figure:
        df = pd.DataFrame(scores, columns=['values'])
        histogram_kwargs = {
            'x': "values",
            'marginal': "box"
        }
        if color:
            histogram_kwargs['color_discrete_sequence'] = [color]
        fig = px.histogram(df, **histogram_kwargs)
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
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
                z=histogram2d,
                x=x_centers, # Centers of bins (x-axis)
                y=y_centers, # Centers of bins (y-axis)
                colorscale=contour_colorscale
            )
        )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig
