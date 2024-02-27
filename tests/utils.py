import torch


# Initialize optimizer, scheduler, and other necessary attributes
def initialize_fit(model):
    model.optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.01, amsgrad=True
    )
    model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode="min", factor=0.1, patience=3
    )
    model.training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_ll": [],
        "validation_accuracy": [],
    }
    model.annealing_factor = 1.0
