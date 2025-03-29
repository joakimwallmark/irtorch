# %%
import torch
from irtorch.models import GradedResponse, MonotoneSpline, MonotonePolynomial, MonotoneNN
from irtorch.estimation_algorithms import JML, MML, AE, VAE
from irtorch.load_dataset import swedish_national_mathematics_1
data = swedish_national_mathematics_1()
model = MonotoneSpline(data)
model2 = MonotonePolynomial(data)
model3 = MonotoneNN(data)
model.plot.plot_item_probabilities(item=1, theta_range=(-6, 6)).show()
model.plot.plot_item_probabilities(item=28, theta_range=(-6, 6)).show()
# model = GradedResponse(data)

# %%
model.fit(train_data=data, algorithm=MML(), device = "cpu")
model.fit(train_data=data, algorithm=JML(), device = "cpu", learning_rate=0.015)
model.fit(train_data=data, algorithm=AE(), device = "cpu")
model.fit(train_data=data, algorithm=VAE(), device = "cpu")
model2.fit(train_data=data, algorithm=MML(), device = "cpu")
model3.fit(train_data=data, algorithm=MML(), device = "cpu")
theta = model.latent_scores(data, theta_estimation="ML", device="cuda", lbfgs_learning_rate=0.1)
theta2 = model2.latent_scores(data, theta_estimation="ML", device="cuda")
theta3 = model3.latent_scores(data, theta_estimation="ML", device="cuda")

# %%
print(model.evaluate.log_likelihood(data, theta))
print(model2.evaluate.log_likelihood(data, theta2))
print(model3.evaluate.log_likelihood(data, theta3))
# %%
print(model.evaluate.predictions(data, theta).mean(axis=0))
print(model2.evaluate.predictions(data, theta2).mean(axis=0))
print(model3.evaluate.predictions(data, theta3).mean(axis=0))

# %%
for item in range(data.shape[1]):
    model.plot.plot_item_probabilities(item=item, theta_range=(-6, 6)).show()

# %%
model.plot.plot_latent_score_distribution(theta).show()
model.plot.plot_item_probabilities(item=3, theta_range=(-6, 6), plot_group_fit=True).show()
model2.plot.plot_item_probabilities(item=item, theta_range=(-6, 6), plot_group_fit=True).show()
