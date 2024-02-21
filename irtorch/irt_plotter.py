import logging
from sklearn.neighbors import KernelDensity
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.irt_scorer import IRTScorer
from irtorch.irt_evaluator import IRTEvaluator
from irtorch.helper_functions import output_to_item_entropy

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

    def plot_training_history(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the training history of the model.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        if all(len(val) == 0 for val in self.algorithm.training_history.values()):
            logging.error("Model has not been trained yet")
            raise AttributeError("Model has not been trained yet")

        measures = {
            "Loss function": {
                "train": "train_loss",
                "validation": "validation_loss",
                "y_label": "Loss",
            }
        }

        existing_measures = [
            m
            for m in measures.values()
            if len(self.algorithm.training_history.get(m.get("train"), [])) > 0
            or len(self.algorithm.training_history.get(m.get("validation"), [])) > 0
        ]

        # If no measures have data, return without creating a plot
        if not existing_measures:
            raise ValueError(
                "None of the selected measures have data available for plotting."
            )

        fig, axs = plt.subplots(
            len(existing_measures), 1, figsize=(12, len(existing_measures) * 3)
        )
        if len(existing_measures) == 1:
            axs = [axs]  # Ensure axs is a list even when only one subplot is created

        for i, measure in enumerate(existing_measures):
            axs[i].grid(True)

            if (
                measure.get("train")
                and len(self.algorithm.training_history[measure["train"]]) > 0
            ):
                axs[i].plot(
                    self.algorithm.training_history[measure["train"]],
                    label="training data",  # Change 'train' to 'training'
                    linewidth=self.linewidth,
                )

                # Add an additional check here to ensure we only plot if the data list is not empty
            if len(self.algorithm.training_history.get(measure["validation"], [])) > 0:
                axs[i].plot(
                    self.algorithm.training_history.get(measure["validation"]),
                    label="validation data",
                    linewidth=self.linewidth,
                )

            axs[i].set_title(
                list(measures.keys())[list(measures.values()).index(measure)]
            )
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel(measure["y_label"])
            axs[i].legend()

        plt.tight_layout()
        return fig, axs

    @torch.inference_mode()
    def plot_latent_score_distribution(
        self,
        scores_to_plot: torch.Tensor = None,
        population_data: torch.Tensor = None,
        latent_variables_to_plot: tuple[int] = (1,),
        kernel_bandwidth = 'scott',
        steps: int = 200,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the distribution of latent scores.

        Parameters
        ----------
        scores_to_plot : torch.Tensor, optional
            If provided, the requested latent variable distributions are plotted directly.
            If None, scores are computed from the population data. (default is None)
        population_data : torch.Tensor, optional
            The data used to compute the latent scores. If None, uses the training data. (default is None)
        latent_variables_to_plot : tuple[int], optional
            The latent dimensions to include in the plot. (default is (1,))
        kernel_bandwidth : float | str, optional
            The bandwidth to use for the kernel density estimate. (default is 'scott' and uses Scott's rule)
        steps : int, optional
            The number of steps to use for computing the kernel density estimate. (default is 200)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the latent_scores method it scores_to_plot is not provided.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
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
                **kwargs
            )
        else:
            scores = scores_to_plot

        return self._kernel_score_distribution_plot(
            scores[:, [i - 1 for i in latent_variables_to_plot]],
            steps,
            kernel_bandwidth
        )

    @torch.inference_mode()
    def plot_item_entropy(
        self,
        item: int,
        scale="bit",
        latent_variables: int = 1,
        steps: int = 1000,
        z_range: tuple[float, float] = (-4, 4),
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the entropy of item responses against latent variables.

        Parameters
        ----------
        item : int
            The item to plot.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : int, optional
            The latent variable dimension to plot. (default is 1)
        steps : int, optional
            The number of steps along the latent variable scale used for entropy evaluation. (default is 1000)
        bit_score_method : str, optional
            The method used for scale value calculation. Only used if scale is 'bit'. (default is 'tanh')
        bit_score_grid_steps : int, optional
            The number of steps used to calculate bit scores. Only used if scale is 'bit'. (default is 300)
        z_range : tuple[float, float], optional
            The range for z-values for plotting. Only used if scale is 'z'. (default is (-4, 4))
        **kwargs
            Additional keyword arguments to pass to the bit score computation.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        z_grid = (
            torch.linspace(z_range[0], z_range[1], steps=steps)
            .repeat(self.model.latent_variables, 1)
            .T
        )

        mean_output = self.model(z_grid)
        item_entropies = output_to_item_entropy(
            mean_output, self.model.modeled_item_responses
        )[:, item - 1]

        if scale == "bit":
            scores_to_plot, _ = self.scorer._bit_scores_from_z(
                z=z_grid,
                **kwargs,
            )
        else:
            scores_to_plot = z_grid[:, latent_variables - 1]

        return self._item_bit_score_plot(scores_to_plot, item_entropies)

    @torch.inference_mode()
    def plot_item_latent_variable_relationships(
        self,
        relationships: torch.Tensor,
        title: str = "Relationships: Items vs. latent variables",
        x_label: str = "Latent variables",
        y_label: str = "Items",
        cmap: str = "inferno",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a heatmap of item-latent variable relationships.

        Parameters
        ----------
        relationships : torch.Tensor
            A tensor of item-latent variable relationships. Typically the returned tensor from expected_item_score_slopes() where each row represents an item and each column represents a latent variable.
        title : str, optional
            The title for the plot. (default is "Relationships: Items vs. latent variables")
        x_label : str, optional
            The label for the X-axis. (default is "Latent variables")
        y_label : str, optional
            The label for the Y-axis. (default is "Items")
        cmap : str, optional
            The matplotlib color map to use for the plot. Use for example "Greys" for black and white. (default is "inferno")

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        relationships = relationships.numpy()

        fig, ax = plt.subplots()
        cax = ax.imshow(relationships, cmap=cmap, aspect='auto')
        fig.colorbar(cax)  # Adds a color bar to the side
        ax.set_xticks(range(relationships.shape[1]))
        ax.set_xticklabels([str(i+1) for i in range(relationships.shape[1])])
        ax.set_yticks(range(relationships.shape[0]))
        ax.set_yticklabels([str(i+1) for i in range(relationships.shape[0])])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        return fig, ax

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
            The latent space variables to plot. (default is (1,))
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
            latent_z_1, latent_z_2 = torch.meshgrid(latent_z_1, latent_z_2)
            z_grid = torch.zeros(latent_z_1.numel(), model_dim)
            z_grid[:, latent_indices[0]] = latent_z_1.flatten()
            z_grid[:, latent_indices[1]] = latent_z_2.flatten()
            z_grid[:, mask] = fixed_zs
            
        prob_matrix = self.model.item_probabilities(z_grid)[:, item - 1, :self.model.modeled_item_responses[item - 1]]

        if scale == "bit":
            scores_to_plot, bit_score_start_z = self.scorer._bit_scores_from_z(
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

    def _kernel_score_distribution_plot(
        self,
        latent_scores: torch.Tensor,
        steps: int = 200,
        kernel_bandwidth = 'scott',
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the kernel density estimate of latent scores.

        Parameters
        ----------
        latent_scores : torch.Tensor
            A tensor of latent scores for which to plot the kernel density estimate.
        steps : int, optional
            The number of steps to use for computing the kernel density estimate. (default is 200)
        kernel_bandwidth : float or {"scott", "silverman"}
            The bandwidth to use for the kernel density estimate. (default is 'scott' and uses Scott's rule)

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        kernel_density = KernelDensity(
            kernel="gaussian", bandwidth=kernel_bandwidth
        ).fit(latent_scores.cpu().numpy())

        # Compute density for mesh of latent variable values
        latent_variables = latent_scores.shape[1]  # number of latent variables
        if latent_variables > 2:
            raise TypeError("Can not plot for more than two latent score dimensions")

        # Create the grid for each variable
        grids = [
            torch.linspace(
                latent_scores[:, i].min(), latent_scores[:, i].max(), steps=steps
            )
            for i in range(latent_variables)
        ]
        # Creating the grid combinations
        mesh = torch.meshgrid(*grids, indexing="ij")
        grid_combinations = torch.stack([t.flatten() for t in mesh]).t()
        # Apply KDE and reshape result
        log_density = torch.from_numpy(
            kernel_density.score_samples(grid_combinations.cpu().numpy())
        )
        density_values = log_density.exp().reshape([steps] * latent_variables)
        latent_scores = latent_scores.cpu()

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlim(min(latent_scores), max(latent_scores))
        if latent_scores.shape[1] == 1:
            ax.hist(latent_scores.squeeze(), density=True, alpha=0.6, zorder=3)
            ax.plot(grids[0], density_values, linewidth=self.linewidth, zorder=4)
            ax.set_xlabel("Latent variable")
            ax.set_ylabel("Density")
            ax.set_title("Latent variable density plot")
            ax.legend()
        else:  # plot for two latent variables
            contour = ax.contourf(
                mesh[0], mesh[1], density_values, levels=100, cmap="RdPu"
            )
            num_dots = min(500, latent_scores.shape[0])
            ax.scatter(
                latent_scores[:num_dots, 0],
                latent_scores[:num_dots, 1],
                color="white",
                s=30,
                edgecolor="black",
                alpha=0.5,
            )
            fig.colorbar(contour, ax=ax)
            ax.set_xlabel("Latent variable 1")
            ax.set_ylabel("Latent variable 2")
            ax.set_title("Latent variable density plot")
        return fig, ax

    def _item_bit_score_plot(
        self,
        latent_scores: torch.Tensor,
        entropies: torch.Tensor,
        x_label: str = "Latent variable",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a plot of item entropies against latent scores.

        Parameters
        ----------
        latent_scores : torch.Tensor
            A tensor of latent scores. These form the X-axis values for the plot.
        entropies : torch.Tensor
            A tensor of item entropies. These form the Y-axis values for the plot.
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        entropies = entropies.cpu().numpy()
        latent_scores = latent_scores.cpu().numpy()
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlim(min(latent_scores), max(latent_scores))
        ax.plot(
            latent_scores,
            entropies,
            linewidth=self.linewidth,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Entropy")
        ax.set_title("Item entropy")
        ax.legend()
        return fig, ax
