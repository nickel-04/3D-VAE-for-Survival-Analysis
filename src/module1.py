
import sys
import os


sys.path.append(os.path.curdir + '/..')
from Dataset.custom_dataset import get_data_loaders

import Dataset
print(Dataset.__name__)

# Example usage
train_loader, test_loader = get_data_loaders(
    root_dir='/home/nikhil/Downloads/tcia_download/NBIA-Download',
    excel_path='/home/nikhil/Downloads/tcia_download/Dataset/crlm.xlsx',
    batch_size=4,
    test_size=0.25
)

# Now you can use train_loader and test_loader in your model
print("len(train_loader)",len(train_loader))
print("len(test_loader)",len(test_loader))



class LatentSpaceGaussian(nn.Module):
    def __init__(self, latent_dim):
        super(LatentSpaceGaussian, self).__init__()
        self.latent_dim = latent_dim
        self.latent_params = 2 * latent_dim  # Two parameters are needed for each Gaussian distribution


    def get_latent_params(self, x):
        x = x.view(-1, 2, self.latent_dim)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        return mu, log_var


    def sample_latent(self, latent_params):
        mu, log_var = latent_params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z


    def kl_loss(self, latent_params):
        mu, log_var = latent_params
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Kullback-Leibler divergence
        return kl / mu.shape[0]

class LogLikelihoodLossWithCensoring(nn.Module):
    def __init__(self, dist_type):
        """
        Initialize the log-likelihood loss with censoring class.
        
        Args:
            dist_type (str): Distribution type to use for survival prediction (currently only supports 'weibull').
        """
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.dist_type = dist_type

    def forward(self, inputs, survival_times, risk_status):
        """
        Compute the log-likelihood loss with censoring.
        
        Args:
            inputs (Tensor): Weibull parameters (alpha and lambda) predicted by the model, shape [batch_size, 2].
            survival_times (Tensor): Actual survival times for each sample, shape [batch_size].
            risk_status (Tensor): Binary censoring indicators (1 if event occurred, 0 if censored), shape [batch_size].

        Returns:
            loss (Tensor): Mean log-likelihood loss with censoring.
        """
        survival_times = torch.squeeze(survival_times)
        risk_status = torch.squeeze(risk_status)

        # Extract Weibull parameters from inputs
        alpha = inputs[:, 0]
        lam = inputs[:, 1]

        # Calculate the survival log-probabilities
        survival_term = -torch.pow(survival_times / lam, alpha)
        
        # Calculate the hazard function where risk_status is 1
        hazard_term = torch.log(alpha / lam) + (alpha - 1) * torch.log(survival_times / lam)

        # Apply the censoring: use hazard term only where risk status is 1 (event occurred)
        log_likelihood_loss = -torch.sum(hazard_term[risk_status > 0.5]) - torch.sum(survival_term)

        # Compute the mean loss across the batch
        loss = log_likelihood_loss / inputs.size(0)
        
        # Ensure finite loss and handle NaNs/INFs
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered, check inputs for stability issues.")

        return loss

import torch
import torch.nn as nn
import numpy as np


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, latent_dim=None):
        super(UNet3D, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder and decoder layers
        self.enc1 = Encoder3D(in_channels, base_channels)
        self.enc2 = Encoder3D(base_channels, base_channels * 2)
        self.enc3 = Encoder3D(base_channels * 2, base_channels * 4)
        self.enc4 = Encoder3D(base_channels * 4, base_channels * 8)
        
        if latent_dim is not None:
            self.latent_space = LatentSpaceGaussian(latent_dim)
            self.latent_mu = nn.Linear(base_channels * 8 * 8 * 8 * 4, self.latent_space.latent_params)
            self.fc_decode = nn.Linear(latent_dim, base_channels * 8 * 8 * 8 * 4)
        
        self.dec4 = Decoder3D(base_channels * 8, base_channels * 4)
        self.dec3 = Decoder3D(base_channels * 4, base_channels * 2)
        self.dec2 = Decoder3D(base_channels * 2, base_channels)
        self.dec1 = Decoder3D(base_channels, out_channels)

        if latent_dim is not None:
            self.time_decoder = TimeDecoder(latent_dim)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        if self.latent_dim is not None:
            enc4_flat = enc4.view(enc4.size(0), -1)
            latent_params = self.latent_mu(enc4_flat)
            mu, log_var = self.latent_space.get_latent_params(latent_params)
            z = self.latent_space.sample_latent((mu, log_var))
            enc4 = self.fc_decode(z)
            enc4 = enc4.view(enc4.size(0), -1, 8, 8, 4)
            kl_divergence = self.latent_space.kl_loss((mu, log_var))
            weibull_params = self.time_decoder(z)
        
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)

        if self.latent_dim is not None:
            return dec1, weibull_params, kl_divergence
        else:
            return dec1

    def predict_time(self, x, just_cov=False, device=torch.device('cpu')):
        """Predicts Weibull time distribution parameters and samples for survival analysis."""
        if not just_cov:
            cov = x.drop(['time', 'event'], axis=1)
        else:
            cov = x.copy()

        # Forward pass to get latent representations and time predictions
        out_params = self.forward(cov.to(device))
        z = out_params[1]  # Assuming 'z' is returned as the second output
        time_params = self.time_decoder(z)

        # Generate samples from time distribution
        time_samples = sample_from_dist(time_params, [self.time_dist])

        return {
            'z': z.detach().cpu().numpy(),
            'time_params': time_params.detach().cpu().numpy(),
            'time_samples': time_samples
        }

    def predict_just_time(self, x):
        """Predicts only the time samples based on the input covariates."""
        return self.predict_time(x, just_cov=True)['time_samples']


    def calculate_risk(self, time_train, x_val, censor_val):
        """Calculate risk CDF at specified time points, compute C-index and IBS with confidence intervals."""
        times = np.unique(time_train)
        out_data = self.predict_time(x_val)
        pred_risk = np.zeros([out_data['time_params'].shape[0], len(times)])

        for sample in range(pred_risk.shape[0]):
            if self.time_dist[0] == 'weibull':
                alpha = out_data['time_params'][sample, 0]
                lam = out_data['time_params'][sample, 1]
                pred_risk[sample, :] = 1 - np.exp(-np.power(times / lam, alpha))
            else:
                raise RuntimeError('Unknown time distribution for risk calculation.')

        # Calculate C-index and IBS along with confidence intervals
        ci, ibs = obtain_c_index(pd.DataFrame(1 - pred_risk.T, index=times), np.array(x_val['time']), censor_val)
        ci_conf_intervals = bern_conf_interval(len(np.array(x_val['time'])), ci)
        ibs_conf_intervals = bern_conf_interval(len(np.array(x_val['time'])), ibs, ibs=True)

        return ci_conf_intervals, ibs_conf_intervals
     

    def fit_epoch(self, data, optimizer, batch_size=64, device=torch.device('cpu')):
        epoch_results = {'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0, 'll_cov_tr': 0.0, 'll_cov_va': 0.0,
                         'll_time_tr': 0.0, 'll_time_va': 0.0, 'ci_va': 0.0, 'ibs_va': 0.0}

        for batch in range(n_batches):
    


    def fit(self, data, train_params):
        training_stats = {'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [], 'll_cov_tr': [], 'll_cov_va': [],
                          'll_time_tr': [], 'll_time_va': [], 'ci_va': [], 'ibs_va': []}

        optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'])

        for epoch in range(train_params['n_epochs']):       
