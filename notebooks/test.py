# PyTorch model and training necessities
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import irtorch.load_dataset as load_dataset
import irtorch.models as models
from irtorch import IRT

torch.set_printoptions(precision=7, sci_mode=False)
device = "cuda"
device = "cpu"
latent_variables = 2
one_hot_encoded = True
negative_latent_variable_item_relationships = True

data = load_dataset.swedish_sat_2022_binary()[:, :80]
data, mc_correct = load_dataset.swedish_sat_2022()
data, mc_correct = load_dataset.swedish_sat_verbal_2022()
data, mc_correct = load_dataset.swedish_sat_quantitative_2022()
writer = SummaryWriter('runs/swesat')
n_cats = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()
train_data = data[:5000, :]
test_data = data[5000:, :]
# model_missing = True
model_missing = False
train_data = train_data[~train_data.isnan().any(dim=1)]
test_data = test_data[~test_data.isnan().any(dim=1)]

ae_model = IRT(
    model = "MMCNN",
    estimation_algorithm="AE",
    latent_variables=latent_variables,
    item_categories=n_cats,
    model_missing=model_missing,
    mc_correct=mc_correct,
    one_hot_encoded=one_hot_encoded
)

ae_model.fit(
    train_data=train_data,
    validation_data=None,
    batch_size=64,
    max_epochs=500,
    learning_rate=0.04,
    learning_rate_update_patience=4,
    learning_rate_updates_before_stopping=5,
    device=device,
)
# ae_model.load_model("notebooks/models/ae_1layer_mmc_2d.pt")

theta = ae_model.latent_scores(train_data)