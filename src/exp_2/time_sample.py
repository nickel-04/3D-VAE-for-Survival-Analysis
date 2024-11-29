
import torch
import torch.nn as nn
import numpy as np

def sample_from_dist(params, distributions):
    """
    Generates samples from specified distribution parameters.
    
    Args:
        params (torch.Tensor): Tensor of distribution parameters, expected shape [batch_size, 2]
                            where params[:, 0] = alpha and params[:, 1] = lambda for Weibull.
        distributions (list): List specifying distribution types, e.g., ['weibull'].

    Returns:
        numpy.ndarray: Array of generated samples from specified distributions.
    """
    samples = []
    for dist in distributions:
        if dist == 'weibull':
            alpha, lam = params[:, 0], params[:, 1]  # Extract Weibull parameters
            alpha = torch.clamp(alpha, min=0.2)  # Prevent alpha from being zero or negative
            lam = torch.clamp(lam, min=0.2)
            # Sample from Weibull distribution
            weibull_samples = lam * (-torch.log(torch.rand_like(alpha))) ** (1 / alpha)
            if (weibull_samples <= 0).any():
                print("Non-positive samples detected:", weibull_samples[weibull_samples <= 0])

            samples.append(weibull_samples.detach().cpu().numpy())  # Convert samples to numpy
        else:
            raise NotImplementedError(f"Unknown distribution type: {dist}")
    
    # Concatenate all samples along the batch dimension
    return np.concatenate(samples, axis=0)
