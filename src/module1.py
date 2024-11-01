
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder3D(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(Encoder3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # Downsample by 2
        )
    
    def forward(self, x):
        return self.encoder(x)
class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3D, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample by 2
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, z):
        return self.decoder(z)

class TimeDecoder(nn.Module):
    def __init__(self, input_dim):
        super(TimeDecoder, self).__init__()
        # Intermediate dimension
        intermediate_dim = input_dim * 2  # You can adjust this multiplier as necessary
        
        # Define a series of Linear layers and ReLUs
        self.layers = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Linear(intermediate_dim // 2, 2)  # Outputs the shape and scale parameters
        )
    
    def forward(self, z):
        return self.layers(z)



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
        
        out_data = {}  # Dictionary to store outputs
        
        if self.latent_dim is not None:
            # Flatten and pass through latent layers
            enc4_flat = enc4.view(enc4.size(0), -1)
            latent_params = self.latent_mu(enc4_flat)
            mu, log_var = self.latent_space.get_latent_params(latent_params)
            z = self.latent_space.sample_latent((mu, log_var))
            
            # Decode the latent sample
            enc4 = self.fc_decode(z)
            enc4 = enc4.view(enc4.size(0), -1, 8, 8, 4)
            
            # Predict Weibull parameters
            weibull_params = self.time_decoder(z)
            
            # Add to out_data dictionary
            out_data['latent_sample'] = z
            out_data['weibull_params'] = weibull_params
            out_data['mu'] = mu
            out_data['log_var'] = log_var
        
        # Decoding path
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)
        
        # Reconstruction output
        out_data['reconstruction'] = dec1
        
        return out_data



    def predict_time(self, images, device=torch.device('cpu')):
        """Predicts Weibull time distribution parameters and samples for survival analysis."""
        self.eval()  # set model to evaluation mode
        images = images.to(device)

        # Forward pass to get output, Weibull params, and KL divergence
        with torch.no_grad():
            output, weibull_params, _ = self.forward(images)

        # Generate samples from time distribution
        time_samples = sample_from_dist(weibull_params, ['weibull'])

        return {
            'weibull_params': weibull_params.detach().cpu().numpy(),
            'time_samples': time_samples
        }


    def predict_just_time(self, images, device=torch.device('cpu')):
        """Predicts only the time samples based on the input covariates."""
        return self.predict_time(images, device=device)['time_samples']

    def calculate_risk(self, time_train, images_val, censor_val, device=torch.device('cpu')):
        """Calculate risk CDF at specified time points, compute C-index and IBS with confidence intervals."""
        self.eval()  # set model to evaluation mode
        times = np.unique(time_train)
        out_data = self.predict_time(images_val, device=device)
        weibull_params = out_data['weibull_params']
        pred_risk = np.zeros([weibull_params.shape[0], len(times)])

        # Compute CDF for risk calculation based on Weibull parameters
        for sample in range(pred_risk.shape[0]):
            alpha = weibull_params[sample, 0]
            lam = weibull_params[sample, 1]
            pred_risk[sample, :] = 1 - np.exp(-np.power(times / lam, alpha))

        # Calculate C-index and IBS with confidence intervals
        ci, ibs = obtain_c_index(pd.DataFrame(1 - pred_risk.T, index=times), np.array(images_val['time']), censor_val)
        ci_conf_intervals = bern_conf_interval(len(np.array(images_val['time'])), ci)
        ibs_conf_intervals = bern_conf_interval(len(np.array(images_val['time'])), ibs, ibs=True)

        return ci_conf_intervals, ibs_conf_intervals

     

    def fit_epoch(self, data, optimizer, batch_size=64, device=torch.device('cpu')):
        epoch_results = {
            'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0,
            'll_cov_tr': 0.0, 'll_cov_va': 0.0, 'll_time_tr': 0.0, 'll_time_va': 0.0,
            'ci_va': 0.0, 'ibs_va': 0.0
        }

        train_loader, val_loader = data  # Expecting DataLoader for training and validation

        # Train epoch
        self.train()  # Set model to training mode
        for batch_idx, (images, time_train, censor_train) in enumerate(train_loader):
            images = images.to(device)
            time_train = time_train.to(device).float()
            censor_train = censor_train.to(device).float()

            # Generate output parameters
            out = self.feed_forward(images)

            # Compute losses
            optimizer.zero_grad()
            loss_kl = self.latent_space.kl_loss(out['latent_params'])
            loss_cov = self.rec_loss(out['cov_params'], images)  # Assuming image reconstruction
            loss_time = self.time_loss(out['time_params'], time_train, censor_train)
            loss = loss_kl + loss_cov + loss_time
            loss.backward()
            optimizer.step()

            # Save data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_cov_tr'] += loss_cov.item()
            epoch_results['ll_time_tr'] += loss_time.item()

        # Validation step
        self.eval()
        with torch.no_grad():
            for images_val, time_val, censor_val in val_loader:
                images_val = images_val.to(device)
                time_val = time_val.to(device).float()
                censor_val = censor_val.to(device).float()

                out = self.feed_forward(images_val)
                loss_kl = self.latent_space.kl_loss(out['latent_params'])
                loss_cov = self.rec_loss(out['cov_params'], images_val)  # Assuming image reconstruction
                loss_time = self.time_loss(out['time_params'], time_val, censor_val)
                loss = loss_kl + loss_cov + loss_time

                # Save data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_cov_va'] += loss_cov.item()
                epoch_results['ll_time_va'] += loss_time.item()

        if self.early_stop:
            self.early_stopper.early_stop(epoch_results['loss_va'])

        return epoch_results
    
    def fit(self, data, train_params):
    training_stats = {
        'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [],
        'll_cov_tr': [], 'll_cov_va': [], 'll_time_tr': [], 'll_time_va': [],
        'ci_va': [], 'ibs_va': []
    }

    optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'])

    epochs_ci = []
    epochs_ibs = []
    train_losses = []
    valid_losses = []
    best_model = None
    best_loss = np.inf
    best_ci = 0.0
    best_ibs = 1.0
    
    # Expecting data to be a tuple of train_loader and val_loader
    train_loader, val_loader = data

    for epoch in range(train_params['n_epochs']):
        self.kl_w = triangle_rate(epoch % 50, n_epochs_total=50, n_epochs_init=25, init_val=1 / 10000, ann_prop=0.5)
        self.cov_w = triangle_rate(epoch % 50, n_epochs_total=50, n_epochs_init=15, init_val=1 / 10000, ann_prop=0.5)

        # Perform training for one epoch
        epoch_results = self.fit_epoch((train_loader, val_loader), optimizer, batch_size=train_params['batch_size'], device=train_params['device'])

        # Calculate metrics
        # Assuming that you have a way to calculate the CI and IBS from the validation dataset
        epoch_results['ci_va'], epoch_results['ibs_va'] = self.calculate_risk(val_loader)

        # Save training stats
        train_losses.append(epoch_results['loss_tr'])
        valid_losses.append(epoch_results['loss_va'])
        epochs_ci.append(epoch_results['ci_va'])
        epochs_ibs.append(epoch_results['ibs_va'])
        for key in epoch_results.keys():
            training_stats[key].append(epoch_results[key])

        if epoch % 50 == 0:
            print('Iteration = ', epoch,
                  '; train loss = ', '{:.2f}'.format(epoch_results['loss_tr']),
                  '; val loss = ', '{:.2f}'.format(epoch_results['loss_va']),
                  '; ll_cov_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_tr'])),
                  '; ll_cov_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_va'])),
                  '; kl_tr = ', '{:.2f}'.format(epoch_results['kl_tr']),
                  '; kl_va = ', '{:.2f}'.format(epoch_results['kl_va']),
                  '; ll_time_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_tr'])),
                  '; ll_time_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_va'])),
                  '; C-index = ', '{:.4f}'.format(np.mean(epoch_results['ci_va'])))

        # Save best model
        if epoch_results['loss_va'] < best_loss:
            best_loss = epoch_results['loss_va']
            best_model = self.state_dict()
            best_ci = epoch_results['ci_va']
            best_ibs = epoch_results['ibs_va']

        if self.early_stop and self.early_stopper.stop:
            print('[INFO] Training early stop')
            break

    # Load the best model after training
    self.load_state_dict(best_model)
    training_stats['ci_va'].append(best_ci)
    training_stats['ibs_va'].append(best_ibs)

    return training_stats
