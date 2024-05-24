# import re
# import torch
# import pytest
# from utils import spearman_correlation
# from irtorch.utils import get_item_categories
# from irtorch import IRT

# @pytest.mark.integration
# def test_model_fit(device, latent_variables, data, data_type, fitting_algorithm):
#     if device == "cuda" and not torch.cuda.is_available():
#         pytest.skip("GPU is not available.")
#     one_hot_encoded=False
#     correct_cat=None
#     if (data_type == "mc"):
#         one_hot_encoded=True
#         with open("tests/datasets/mc_correct.txt", "r") as file:
#             correct_cat = file.read().replace("\n", "")
#         correct_cat = [int(num) for num in correct_cat]
#         n_cats = [4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5]
#     else:
#         n_cats = get_item_categories(data)

#     torch.manual_seed(125)
#     model = IRT(
#         estimation_algorithm=fitting_algorithm,
#         latent_variables=latent_variables,
#         item_categories=n_cats,
#         one_hot_encoded=one_hot_encoded,
#         mc_correct=correct_cat
#     )
#     initial_parameter_dict = {k: v.clone() for k, v in model.model.state_dict().items()}
#     model.fit(
#         train_data=data,
#         device=device
#     )
    
#     # Save models for other tests
#     # model.save_model(f"tests/fitted_models/{model_type}_latent_variables{latent_variables}_{data_type}_{device}.pt")

#     for name, param in model.model.state_dict().items():
#         if "free" not in name and "missing_categories" not in name and not re.match(r"negation_dim\d+\.weight", name):
#             assert not torch.equal(initial_parameter_dict[name], param), f"Parameter {name} was not updated."

# @pytest.mark.integration
# def test_latent_scores(model: IRT, data, latent_variables):
#     theta_nn_scores = model.scorer.latent_scores(data, theta_estimation="NN")
#     theta_ml_scores = model.scorer.latent_scores(data, theta_estimation="ML")
#     multi_bit_score_nn_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="NN", bit_score_one_dimensional=False)
#     multi_bit_score_ml_scores = model.scorer.latent_scores(data, scale="bit", theta = theta_ml_scores, theta_estimation="ML", bit_score_population_theta=theta_ml_scores, bit_score_one_dimensional=False)
#     one_bit_score_nn_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="NN", bit_score_one_dimensional=True)
#     one_bit_score_ml_scores = model.scorer.latent_scores(data, scale="bit", theta = theta_ml_scores, theta_estimation="ML", bit_score_population_theta=theta_ml_scores, bit_score_one_dimensional=True)

#     # TODO Add map and EAP again
#     # multi_bit_score_map_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="MAP", bit_score_one_dimensional=False)
#     # multi_bit_score_eap_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="EAP", bit_score_one_dimensional=False)
#     # theta_map_scores = model.scorer.latent_scores(data, theta_estimation="MAP")
#     # theta_eap_scores = model.scorer.latent_scores(data, theta_estimation="EAP")
#     # one_bit_score_map_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="MAP", bit_score_one_dimensional=True)
#     # one_bit_score_eap_scores = model.scorer.latent_scores(data, scale="bit", theta_estimation="EAP", bit_score_one_dimensional=True)

#     assert theta_nn_scores.shape == torch.Size([data.shape[0], latent_variables])
#     assert theta_ml_scores.shape == torch.Size([data.shape[0], latent_variables])
#     # assert theta_map_scores.shape == torch.Size([data.shape[0], latent_variables])
#     # assert theta_eap_scores.shape == torch.Size([data.shape[0], latent_variables])
#     assert multi_bit_score_nn_scores.shape == torch.Size([data.shape[0], latent_variables])
#     assert multi_bit_score_ml_scores.shape == torch.Size([data.shape[0], latent_variables])
#     # assert multi_bit_score_map_scores.shape == torch.Size([data.shape[0], latent_variables])
#     # assert multi_bit_score_eap_scores.shape == torch.Size([data.shape[0], latent_variables])
#     assert one_bit_score_nn_scores.shape == torch.Size([data.shape[0], 1])
#     assert one_bit_score_ml_scores.shape == torch.Size([data.shape[0], 1])
#     # assert one_bit_score_map_scores.shape == torch.Size([data.shape[0], 1])
#     # assert one_bit_score_eap_scores.shape == torch.Size([data.shape[0], 1])

#     for latent_var in range(latent_variables):
#         # Remove duplicates and check spearman correlation
#         # and use is close and not equal two for when we have an uneven amount of perfectly negatively correlated values
#         sorted_scores, indices = multi_bit_score_nn_scores[:, latent_var].sort()
#         unique_a, counts = sorted_scores.unique(sorted=True, return_counts = True)
#         first_occurrences = torch.cumsum(counts, dim=0) - counts
#         unique_b = theta_nn_scores[indices, latent_var][first_occurrences]
#         corr = spearman_correlation(unique_a, unique_b)
#         assert (torch.isclose(corr, torch.tensor(1.0), atol = 1e-5) or torch.isclose(corr, torch.tensor(-1.0), atol = 1e-5))
#         sorted_scores, indices = multi_bit_score_ml_scores[:,latent_var].sort()
#         unique_a, counts = sorted_scores.unique(sorted=True, return_counts = True)
#         first_occurrences = torch.cumsum(counts, dim=0) - counts
#         unique_b = theta_ml_scores[indices, latent_var][first_occurrences]
#         corr = spearman_correlation(unique_a, unique_b)
#         assert (torch.isclose(corr, torch.tensor(1.0), atol = 1e-5) or torch.isclose(corr, torch.tensor(-1.0), atol = 1e-5))
#         # sorted_scores, indices = multi_bit_score_map_scores[:,latent_var].sort()
#         # unique_a, counts = sorted_scores.unique(sorted=True, return_counts = True)
#         # first_occurrences = torch.cumsum(counts, dim=0) - counts
#         # unique_b = theta_map_scores[indices, latent_var][first_occurrences]
#         # corr = spearman_correlation(unique_a, unique_b)
#         # assert (torch.isclose(corr, torch.tensor(1.0), atol = 1e-5) or torch.isclose(corr, torch.tensor(-1.0), atol = 1e-5))
#         # sorted_scores, indices = multi_bit_score_eap_scores[:,latent_var].sort()
#         # unique_a, counts = sorted_scores.unique(sorted=True, return_counts = True)
#         # first_occurrences = torch.cumsum(counts, dim=0) - counts
#         # unique_b = theta_eap_scores[indices, latent_var][first_occurrences]
#         # corr = spearman_correlation(unique_a, unique_b)
#         # assert (torch.isclose(corr, torch.tensor(1.0), atol = 1e-5) or torch.isclose(corr, torch.tensor(-1.0), atol = 1e-5))

#     if latent_variables == 1:
#         assert torch.allclose(one_bit_score_nn_scores, multi_bit_score_nn_scores, atol=1e-01)
#         assert torch.allclose(one_bit_score_ml_scores, multi_bit_score_ml_scores, atol=1e-01)
#         # assert torch.allclose(one_bit_score_map_scores, multi_bit_score_map_scores, atol=1e-01)
#         # assert torch.allclose(one_bit_score_eap_scores, multi_bit_score_eap_scores, atol=1e-01)

# @pytest.mark.integration
# def test_plot_item_probabilities(model: IRT, latent_variables):
#     model.plot_item_probabilities(10, scale = "theta")
#     model.plot_item_probabilities(12, scale = "theta", group_fit_groups=8, plot_group_fit=True)
#     model.plot_item_probabilities(12, scale = "entropy", steps=40, group_fit_groups=8, plot_group_fit=True)
#     model.plot_item_probabilities(12, scale = "entropy", steps=40)
#     if latent_variables > 1:
#         model.plot_item_probabilities(10, scale = "theta", latent_variables=(2, ))
#         model.plot_item_probabilities(10, scale = "theta", latent_variables=(1, 2))
#         if latent_variables == 2:
#             with pytest.raises(TypeError):
#                 model.plot_item_probabilities(10, scale = "theta", latent_variables=(2, ), fixed_thetas=(1, 0.5))
#         else:
#             model.plot_item_probabilities(10, scale = "theta", latent_variables=(1, 2), fixed_thetas=(1, ))

#     # TODO add bit scores plots

# @pytest.mark.integration
# def test_bit_score_starting_theta(model: IRT):
#     guessing_probabilities = [0.5] * len(model.scorer.algorithm.model.modeled_item_responses)

#     # Test with no guessing (or default guessing for multiple choice)
#     starting_theta = model.scorer.bit_score_starting_theta(guessing_iterations=200)
#     assert starting_theta.shape == (1, model.model.latent_variables)

#     # Test with guessing for dichotomous items
#     starting_theta = model.scorer.bit_score_starting_theta(guessing_probabilities=guessing_probabilities, guessing_iterations=200)
#     assert starting_theta.shape == (1, model.model.latent_variables)

#     # Test with guessing and MC correct
#     model.scorer.algorithm.model.mc_correct = [1] * len(model.scorer.algorithm.model.modeled_item_responses)
#     starting_theta = model.scorer.bit_score_starting_theta(guessing_probabilities=guessing_probabilities, guessing_iterations=200)
#     assert starting_theta.shape == (1, model.model.latent_variables)
#     model.scorer.algorithm.model.mc_correct = None

# @pytest.mark.integration
# def test_probability_gradients(model: IRT, latent_variables):
#     # This makes sure we can run torch.vmap inside probability_gradients, as it is not compatible with some tensor operations
#     theta = torch.randn((40, latent_variables))
#     gradients = model.model.probability_gradients(theta)
#     assert gradients.shape == torch.Size([theta.shape[0], model.model.items, model.model.max_item_responses, model.model.latent_variables])
