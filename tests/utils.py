import torch

def spearman_correlation(x, y):
    # Rank the values
    rank_x = torch.argsort(torch.argsort(x)).float()
    rank_y = torch.argsort(torch.argsort(y)).float()
    
    # Compute the difference between ranks
    d = rank_x - rank_y
    
    # Compute the Spearman rank correlation coefficient
    n = x.shape[0]
    rs = 1 - 6 * torch.sum(d**2) / (n * (n**2 - 1))
    
    return rs

def pearson_correlation(x, y):
    # Compute the means
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    # Compute the covariance
    cov = torch.mean((x - mean_x) * (y - mean_y))
    
    # Compute the standard deviations
    std_x = torch.std(x)
    std_y = torch.std(y)
    
    # Compute the correlation coefficient
    r = cov / (std_x * std_y)
    
    return r

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
