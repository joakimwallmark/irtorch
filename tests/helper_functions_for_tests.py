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