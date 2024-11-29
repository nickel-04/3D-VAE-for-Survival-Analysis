import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataloader.dataloader import get_data_loaders
from pycox.evaluation import EvalSurv
from statsmodels.stats.proportion import proportion_confint
from exp_2.model_exp2 import Encoder3D, Decoder3D, TimeDecoder
from exp_2.loss_exp2 import LogLikelihoodLossWithCensoring, contrastive_loss
from exp_2.metric_exp2 import ci_ibs
from exp_2.plot import plot_axial_view ,visualize_last_latent_space, visualize_first_latent_space, plot_individual_losses, plot_ci_ibs
from exp_2.time_sample import sample_from_dist

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, latent_dim=None, save_axial_filename=None,num_points=10):
        super(UNet3D, self).__init__()
        
        self.latent_dim = latent_dim
        self.rec_loss = nn.functional.binary_cross_entropy
        self.time_loss = LogLikelihoodLossWithCensoring()
        self.plot_axial_view = plot_axial_view 
        self.contrastive_loss = contrastive_loss
        self.sample_from_dist = sample_from_dist
        self.ci_ibs = ci_ibs()

        self.save_axial_filename = save_axial_filename if save_axial_filename else 'output/axial_view_plot_2.png'
        # Encoder layers
        self.enc1 = Encoder3D(in_channels, base_channels)
        self.enc2 = Encoder3D(base_channels, base_channels * 2)
        self.enc3 = Encoder3D(base_channels * 2, base_channels * 4)
        self.enc4 = Encoder3D(base_channels * 4, base_channels * 8)
        
        if latent_dim is not None:
            # Separate layers for mu and logvar
            self.mu = nn.Linear(base_channels * 8 * 8 * 8 * 4, latent_dim)
            self.logvar = nn.Linear(base_channels * 8 * 8 * 8 * 4, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, base_channels * 8 * 8 * 8 * 4)  # Reshape latent back to decoder input size
        
        # Decoder layers
        self.dec4 = Decoder3D(base_channels * 8, base_channels * 4)
        self.dec3 = Decoder3D(base_channels * 4, base_channels * 2)
        self.dec2 = Decoder3D(base_channels * 2, base_channels)
        self.dec1 = Decoder3D(base_channels, out_channels)

        if latent_dim is not None:
            self.time_decoder = TimeDecoder(latent_dim, num_points)
    
    def forward(self, x):
            batch_size, _, H, W, D = x.shape 
            out_data = {}  # Dictionary to store outputs
            
            # Encoding path
            enc1 = self.enc1(x)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)
            
            if self.latent_dim is not None:
                # Flatten for mu and logvar calculations
                enc4_flat = enc4.view(enc4.size(0), -1)
                
                # Calculate `mu` and `log_var` using separate layers
                mu = self.mu(enc4_flat)
                log_var = self.logvar(enc4_flat)
                
                # Sample from latent space using reparameterization trick
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Decode the latent sample
                enc4_decoded = self.fc_decode(z).view(batch_size, -1, 8, 8, 4)
                
                # Predict survival and hazard parameters across time points
                surv, haz = self.time_decoder(z)
                
                # Calculate KL divergence
                kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Add outputs to the dictionary
                out_data['latent_sample'] = z
                out_data['mu'] = mu
                out_data['log_var'] = log_var
                out_data['surv'] = surv
                out_data['haz'] = haz
                out_data['kl_divergence'] = kl_divergence
                enc4 = enc4_decoded  # Use the decoded latent sample for further decoding
            
            # Decoding path
            dec4 = self.dec4(enc4)
            dec3 = self.dec3(dec4 + enc3)
            dec2 = self.dec2(dec3 + enc2)
            dec1 = self.dec1(dec2 + enc1)
            
            # Reconstruction output
            out_data['reconstruction'] = torch.sigmoid(dec1)
            
            return out_data

    def fit_epoch(self, train_loader, test_loader, optimizer, epoch, num_epochs, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        epoch_results = {
            'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0,
            'll_img_tr': 0.0, 'll_img_va': 0.0, 'll_time_tr': 0.0, 'll_time_va': 0.0,
            'ci_tr': 0.0, 'ibs_tr': 0.0, 'ci_va': 0.0, 'ibs_va': 0.0,
            # Saving the latent and event data for both train and validation
            'train_latents_first': [], 'train_events_first': [], 'train_times_first': [],
            'val_latents_first': [], 'val_events_first': [], 'val_times_first': [],
            'train_latents_last': [], 'train_events_last': [], 'train_times_last': [],
            'val_latents_last': [], 'val_events_last': [], 'val_times_last': []
        }

        # Initialize lists to collect training data
        all_train_survival_times = []
        all_train_surv = []  # We will collect the time_decoder_params for training set
        all_train_haz = [] # We will collect the time_params for training set
        all_train_events = []
        all_train_latent =[]
        # Train epoch
        self.train()
        for batch in train_loader:
            img_train_batch, time_train_batch, censor_train_batch = batch

            img_train_batch = img_train_batch.to(device)
            time_train_batch = time_train_batch.to(device)
            censor_train_batch = censor_train_batch.to(device)

            # Generate output params (latent sample and time parameters)
            out = self.forward(img_train_batch)

            # Compute losses
            optimizer.zero_grad()
            loss_kl = out['kl_divergence']
            loss_img = self.rec_loss(out['reconstruction'], img_train_batch, reduction='mean')
            loss_time = self.time_loss(out['surv'], time_train_batch, censor_train_batch)
            contrastive_loss_value = self.contrastive_loss(out['latent_sample'], censor_train_batch)
            loss = loss_kl + loss_img + loss_time + contrastive_loss_value

            loss.backward()
            optimizer.step()

            # Save training data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_img_tr'] += loss_img.item()
            epoch_results['ll_time_tr'] += loss_time.item()

            # Collect predictions and ground truth for the entire training set
            all_train_survival_times.append(time_train_batch)
            all_train_surv.append(out['surv'])
            all_train_haz.append(out['haz']) # Collect time_params directly
            all_train_events.append(censor_train_batch)
            all_train_latent.append(out['latent_sample'])

        # After looping through the entire train_loader, concatenate all data for C-index and IBS calculation
        all_train_survival_times = torch.cat(all_train_survival_times, dim=0).to(device)
        all_train_surv = torch.cat(all_train_surv, dim=0).to(device)  # Concatenate time_params
        all_train_haz = torch.cat(all_train_haz, dim=0).to(device)  # Concatenate time_params
        all_train_events = torch.cat(all_train_events, dim=0).to(device)
        all_train_latent = torch.cat(all_train_latent, dim=0).to(device)

        # Calculate the C-index and IBS for the training set
        epoch_results['ci_tr'], epoch_results['ibs_tr'] = self.ci_ibs.calculate_risk(
            all_train_survival_times,
            all_train_surv ,  # Pass the concatenated time_params
            all_train_events, 
            device=device  # Ensure the device is passed to the calculate_risk method
        )

        # Initialize lists to collect validation data
        all_val_survival_times = []
        all_val_surv = []  # We will collect the
        all_val_haz = []  # We will collect the time_params for validation set
        all_val_events = []
        all_val_latent =[]
        # Validation step
        self.eval()
        with torch.no_grad():
            for batch in test_loader:
                img_val, time_val, censor_val = batch

                img_val = img_val.to(device)
                time_val = time_val.to(device)
                censor_val = censor_val.to(device)

                # Generate output params (latent sample and time parameters)
                out = self.forward(img_val)

                # Compute losses
                loss_kl = out['kl_divergence']
                loss_img = self.rec_loss(out['reconstruction'], img_val, reduction='mean')
                loss_time = self.time_loss(out['surv'], time_val, censor_val)
                contrastive_loss_value = self.contrastive_loss(out['latent_sample'], censor_val)
                loss = loss_kl + loss_img + loss_time + contrastive_loss_value

                # Save validation data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_img_va'] += loss_img.item()
                epoch_results['ll_time_va'] += loss_time.item()

                # Collect predictions and ground truth for the entire validation set
                all_val_survival_times.append(time_val)
                all_val_surv.append(out['surv'])
                all_val_haz.append(out['haz'])  # Collect time_params directly
                all_val_events.append(censor_val)
                all_val_latent.append(out['latent_sample'])

                # For the last epoch, only use the last batch for visualization
                if epoch == num_epochs - 1 and len(all_val_latent) == len(test_loader):  # Last Epoch and batch only
                    # Call the plot_axial_view function for the last batch from the validation data
                    self.plot_axial_view(img_val, out['reconstruction'], title=f"Validation Reconstruction - Epoch {epoch + 1}, save_filename=self.save_filename")


        # After looping through the entire test_loader, concatenate all data for C-index and IBS calculation
        all_val_survival_times = torch.cat(all_val_survival_times, dim=0).to(device)
        all_val_surv = torch.cat(all_val_surv, dim=0).to(device)  # Concatenate time_params
        all_val_haz = torch.cat(all_val_haz, dim=0).to(device)   # Concatenate time_params
        all_val_events = torch.cat(all_val_events, dim=0).to(device)
        all_val_latent = torch.cat(all_val_latent, dim=0).to(device)

        # Calculate the C-index and IBS for the validation set
        epoch_results['ci_va'], epoch_results['ibs_va'] = self.ci_ibs.calculate_risk(
            all_val_survival_times,
            all_val_surv,  # Pass the concatenated time_params
            all_val_events,
            device=device  # Ensure the device is passed to the calculate_risk method
        )

        if epoch == 1:
            epoch_results['train_latents_first'] = all_train_latent
            epoch_results['train_events_first'] = all_train_events
            epoch_results['train_times_first'] = all_train_survival_times
            epoch_results['val_latents_first'] = all_val_latent
            epoch_results['val_events_first'] = all_val_events
            epoch_results['val_times_first'] = all_val_survival_times
            # Only collect latent space data during the final epoch

        if epoch == num_epochs - 1:
            epoch_results['train_latents_last'] = all_train_latent
            epoch_results['train_events_last'] = all_train_events
            epoch_results['train_times_last'] = all_train_survival_times
            epoch_results['val_latents_last'] = all_val_latent
            epoch_results['val_events_last'] = all_val_events
            epoch_results['val_times_last'] = all_val_survival_times

            self.plot_axial_view(img_val, out['reconstruction'],
                                  title="Validation Reconstruction - Epoch {epoch + 1}")

        return epoch_results


    def fit(self, train_loader, test_loader, optimizer, num_epochs, device=None, save_path= None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # To keep track of the performance metrics
        training_stats = {
            'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [],
            'll_img_tr': [], 'll_img_va': [], 'll_time_tr': [], 'll_time_va': [],
            'ci_tr': [], 'ci_va': [], 'ibs_tr': [], 'ibs_va': [], 
            # Saving the latent and event data for both train and validation
            'train_latents_first': [], 'train_events_first': [], 'train_times_first': [],
            'val_latents_first': [], 'val_events_first': [], 'val_times_first': [],
            'train_latents_last': [], 'train_events_last': [], 'train_times_last': [],
            'val_latents_last': [], 'val_events_last': [], 'val_times_last': []
        }

        best_model_state = None
        best_ci_va = -float('inf')  # C-Index should be maximized, start with very low value
        best_ibs_va = float('inf')  # IBS should be minimized, start with very high value
        best_epoch = 0

        # Loop over the number of epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Fit for one epoch
            epoch_results = self.fit_epoch(train_loader, test_loader, optimizer, epoch, num_epochs, device)
            
            # Append results to training_stats
            training_stats['loss_tr'].append(epoch_results['loss_tr'])
            training_stats['loss_va'].append(epoch_results['loss_va'])
            training_stats['kl_tr'].append(epoch_results['kl_tr'])
            training_stats['kl_va'].append(epoch_results['kl_va'])
            training_stats['ll_img_tr'].append(epoch_results['ll_img_tr'])
            training_stats['ll_img_va'].append(epoch_results['ll_img_va'])
            training_stats['ll_time_tr'].append(epoch_results['ll_time_tr'])
            training_stats['ll_time_va'].append(epoch_results['ll_time_va'])
            training_stats['ci_tr'].append(epoch_results['ci_tr'])
            training_stats['ci_va'].append(epoch_results['ci_va'])
            training_stats['ibs_tr'].append(epoch_results['ibs_tr'])
            training_stats['ibs_va'].append(epoch_results['ibs_va'])

            # Collect latent and event data only after the last epoch
            if epoch == 1:
                training_stats['train_latents_first'] = epoch_results['train_latents_first']
                training_stats['train_events_first'] = epoch_results['train_events_first'] 
                training_stats['train_times_first'] = epoch_results['train_times_first']
                training_stats['val_latents_first']  = epoch_results['val_latents_first'] 
                training_stats['val_events_first']  = epoch_results['val_events_first'] 
                training_stats['val_times_first']  = epoch_results['val_times_first'] 
                # Only collect latent space data during the final epoch

            if epoch == num_epochs - 1:
                training_stats['train_latents_last']= epoch_results['train_latents_last'] 
                training_stats['train_events_last'] = epoch_results['train_events_last'] 
                training_stats['train_times_last'] = epoch_results['train_times_last'] 
                training_stats['val_latents_last'] = epoch_results['val_latents_last'] 
                training_stats['val_events_last'] = epoch_results['val_events_last'] 
                training_stats['val_times_last'] = epoch_results['val_times_last'] 

            # Print the results for this epoch
            print(f"Train Loss: {epoch_results['loss_tr']:.4f}, "
                f"Validation Loss: {epoch_results['loss_va']:.4f}")
            print(f"Train C-Index: {epoch_results['ci_tr'][1]:.4f}, "
                f"Validation C-Index: {epoch_results['ci_va'][1]:.4f}")
            print(f"Train IBS: {epoch_results['ibs_tr'][1]:.4f}, "
                f"Validation IBS: {epoch_results['ibs_va'][1]:.4f}")
            
            # Save the best model based on validation ci nd ibs
            if (epoch_results['ci_va'][1] > best_ci_va) and (epoch_results['ibs_va'][1] < best_ibs_va):
                best_ci_va = epoch_results['ci_va'][1]
                best_ibs_va = epoch_results['ibs_va'][1]
                best_model_state = self.state_dict()  # Save model state_dict
                best_epoch = epoch  # Track the best epoch
                print(f"New best model found at epoch {epoch + 1} with C-Index: {best_ci_va:.4f} and IBS: {best_ibs_va:.4f}")

        # Save the final model, optimizer state, and training stats after all epochs
        torch.save({
            'epoch': best_epoch,
            'best_ci' :best_ci_va,
            'best_ibs': best_ibs_va,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'training_stats': training_stats
        }, save_path)

        print(f"Training completed. Model saved at {save_path}")
        
        return training_stats

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", message=".*is_monotonic.*", category=DeprecationWarning)
    import warnings
    warnings.filterwarnings("ignore", message=".*Unknown extension is not supported.*")
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    torch.manual_seed(42)
    # Set device for training (GPU if available, otherwise CPU)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Initialize model and move it to the specified device
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, latent_dim=256, save_axial_filename='output/axial_view_plot_2.1.png').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loader, test_loader = get_data_loaders(
        root_dir='Converted_NIfTI',
        excel_path='input/crlm.xlsx',
        batch_size=8,
        test_size=0.30 
    )
    # print("len(train_loader):", len(train_loader))
    # print("len(test_loader):", len(test_loader))

    # Call the fit method with training and validation data loaders
    training_stats = model.fit(train_loader, test_loader, optimizer, num_epochs=125, 
                            device = device, save_path = 'output/final_model_2.1.pth') #save_axial_filename='output/axial_view_plot_1.png'


    plot_individual_losses(training_stats, filename="output/loss_plot_2.1.png")
    plot_ci_ibs(training_stats, filename="output/metrics_plot_2.1.png")

    save_path = 'output/latent_last_space_visualization_2.1.png'
    visualize_last_latent_space(training_stats, save_path)

    # For the first latent states:
    save_path_first = 'output/first_latent_space_visualization_2.1.png'
    visualize_first_latent_space(training_stats, save_path_first)
