# visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_axial_view(image, reconstruction, num_samples=3, title="Axial View", save_axial_filename=None):
    """
    Plots the axial view (middle slice along the Z-axis) for random samples of the 3D image and reconstruction.
    Optionally, saves the plot to a specified file path.

    Args:
        image (torch.Tensor): The original 3D image tensor of shape [B, C, H, W, D].
        reconstruction (torch.Tensor): The reconstructed 3D image tensor of shape [B, C, H, W, D].
        num_samples (int): Number of random samples to plot.
        title (str): Title for the plot.
        save_filename (str): Path to save the plot. If None, the plot won't be saved.
    """
    # Print shapes for debugging
    print("Image tensor shape:", image.shape)
    print("Reconstruction tensor shape:", reconstruction.shape)

    batch_size = image.shape[0]
    fig, axs = plt.subplots(2, num_samples, figsize=(5 * num_samples, 10))
    
    for i in range(num_samples):
        # Choose a random sample from the batch
        idx = random.randint(0, batch_size - 1)

        # Get the original and reconstructed images
        original = image[idx].detach().cpu().numpy()  # Shape: [C, H, W, D]
        reconstructed = reconstruction[idx].detach().cpu().numpy()  # Shape: [C, H, W, D]

        # Remove channel dimension if present
        if original.shape[0] == 1:
            original = original[0]  # Shape: [H, W, D]
        if reconstructed.shape[0] == 1:
            reconstructed = reconstructed[0]  # Shape: [H, W, D]

        # Print the shapes of individual samples
        print(f"Sample {i+1} - Original shape: {original.shape}, Reconstructed shape: {reconstructed.shape}")

        # Get the middle slice along the Z-axis
        middle_slice_idx = original.shape[2] // 2
        original_axial = np.rot90(original[:, :, middle_slice_idx], k=-1)  # Rotate for proper orientation
        reconstructed_axial = np.rot90(reconstructed[:, :, middle_slice_idx], k=-1)

        # Plot the original axial view
        axs[0, i].imshow(original_axial, cmap='gray')
        axs[0, i].set_title(f"Original Sample {i+1} (Z={middle_slice_idx})")
        axs[0, i].axis('off')

        # Plot the reconstructed axial view
        axs[1, i].imshow(reconstructed_axial, cmap='gray')
        axs[1, i].set_title(f"Reconstructed Sample {i+1} (Z={middle_slice_idx})")
        axs[1, i].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save the plot if save_filename is provided
    if save_axial_filename:
        save_dir = os.path.dirname(save_axial_filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_axial_filename)
        print(f"Plot saved at {save_axial_filename}")

    plt.show()


import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_last_latent_space(training_stats, save_directory):
    # Retrieve the last latent and event data
    latents_train_last = training_stats['train_latents_last']
    latents_val_last = training_stats['val_latents_last']
    events_train_last = training_stats['train_events_last']
    events_val_last = training_stats['val_events_last']
    times_train_last = training_stats['train_times_last']
    times_val_last = training_stats['val_times_last']

    # Move the data to CPU and convert to NumPy arrays for visualization
    latents_train_last = latents_train_last.detach().cpu().numpy() if isinstance(latents_train_last, torch.Tensor) else latents_train_last
    latents_val_last = latents_val_last.detach().cpu().numpy() if isinstance(latents_val_last, torch.Tensor) else latents_val_last
    events_train_last = events_train_last.detach().cpu().numpy() if isinstance(events_train_last, torch.Tensor) else events_train_last
    events_val_last = events_val_last.detach().cpu().numpy() if isinstance(events_val_last, torch.Tensor) else events_val_last
    times_train_last = times_train_last.detach().cpu().numpy() if isinstance(times_train_last, torch.Tensor) else times_train_last
    times_val_last = times_val_last.detach().cpu().numpy() if isinstance(times_val_last, torch.Tensor) else times_val_last

    # Perform PCA to reduce the latent space to 2D for visualization
    pca = PCA(n_components=2)
    latent_train_2d = pca.fit_transform(latents_train_last)
    latent_val_2d = pca.transform(latents_val_last)

    # Create a figure with four subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot latent_train_last vs events_train_last
    scatter_train_events = axs[0, 0].scatter(latent_train_2d[:, 0], latent_train_2d[:, 1], c=events_train_last, cmap='coolwarm', alpha=0.7)
    axs[0, 0].set_title('Latent Space vs Events - Training Data')
    axs[0, 0].set_xlabel('Latent Component 1')
    axs[0, 0].set_ylabel('Latent Component 2')
    axs[0, 0].legend(*scatter_train_events.legend_elements(), title="Events")

    # Plot latent_train_last vs times_train_last
    scatter_train_times = axs[0, 1].scatter(latent_train_2d[:, 0], latent_train_2d[:, 1], c=times_train_last, cmap='coolwarm', alpha=0.7)
    axs[0, 1].set_title('Latent Space vs Times - Training Data')
    axs[0, 1].set_xlabel('Latent Component 1')
    axs[0, 1].set_ylabel('Latent Component 2')
    axs[0, 1].legend(*scatter_train_times.legend_elements(), title="Time")

    # Plot latent_val_last vs events_val_last
    scatter_val_events = axs[1, 0].scatter(latent_val_2d[:, 0], latent_val_2d[:, 1], c=events_val_last, cmap='coolwarm', alpha=0.7)
    axs[1, 0].set_title('Latent Space vs Events - Validation Data')
    axs[1, 0].set_xlabel('Latent Component 1')
    axs[1, 0].set_ylabel('Latent Component 2')
    axs[1, 0].legend(*scatter_val_events.legend_elements(), title="Events")

    # Plot latent_val_last vs times_val_last
    scatter_val_times = axs[1, 1].scatter(latent_val_2d[:, 0], latent_val_2d[:, 1], c=times_val_last, cmap='coolwarm', alpha=0.7)
    axs[1, 1].set_title('Latent Space vs Times - Validation Data')
    axs[1, 1].set_xlabel('Latent Component 1')
    axs[1, 1].set_ylabel('Latent Component 2')
    axs[1, 1].legend(*scatter_val_times.legend_elements(), title="Time")

    # Show colorbar for event/time labels
    fig.colorbar(scatter_train_events, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1, label='Event/Time')

    # Create the directory if it doesn't exist (if you want to ensure the directory exists)
    save_dir = os.path.dirname(save_directory)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot to the specified directory (including filename)
    plt.tight_layout()
    plt.savefig(save_directory)
    print(f"Plot saved at {save_directory}")

    plt.show()


def visualize_first_latent_space(training_stats, save_directory):
    # Retrieve the first latent and event data
    latents_train_first = training_stats['train_latents_first']
    latents_val_first = training_stats['val_latents_first']
    events_train_first = training_stats['train_events_first']
    events_val_first = training_stats['val_events_first']
    times_train_first = training_stats['train_times_first']
    times_val_first = training_stats['val_times_first']

    # Move the data to CPU and convert to NumPy arrays for visualization
    latents_train_first = latents_train_first.detach().cpu().numpy() if isinstance(latents_train_first, torch.Tensor) else latents_train_first
    latents_val_first = latents_val_first.detach().cpu().numpy() if isinstance(latents_val_first, torch.Tensor) else latents_val_first
    events_train_first = events_train_first.detach().cpu().numpy() if isinstance(events_train_first, torch.Tensor) else events_train_first
    events_val_first = events_val_first.detach().cpu().numpy() if isinstance(events_val_first, torch.Tensor) else events_val_first
    times_train_first = times_train_first.detach().cpu().numpy() if isinstance(times_train_first, torch.Tensor) else times_train_first
    times_val_first = times_val_first.detach().cpu().numpy() if isinstance(times_val_first, torch.Tensor) else times_val_first

    # Perform PCA to reduce the latent space to 2D for visualization
    pca = PCA(n_components=2)
    latent_train_2d = pca.fit_transform(latents_train_first)
    latent_val_2d = pca.transform(latents_val_first)

    # Create a figure with four subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot latent_train_first vs events_train_first
    scatter_train_events = axs[0, 0].scatter(latent_train_2d[:, 0], latent_train_2d[:, 1], c=events_train_first, cmap='coolwarm', alpha=0.7)
    axs[0, 0].set_title('Latent Space vs Events - Training Data (First Latent)')
    axs[0, 0].set_xlabel('Latent Component 1')
    axs[0, 0].set_ylabel('Latent Component 2')
    axs[0, 0].legend(*scatter_train_events.legend_elements(), title="Events")

    # Plot latent_train_first vs times_train_first
    scatter_train_times = axs[0, 1].scatter(latent_train_2d[:, 0], latent_train_2d[:, 1], c=times_train_first, cmap='coolwarm', alpha=0.7)
    axs[0, 1].set_title('Latent Space vs Times - Training Data (First Latent)')
    axs[0, 1].set_xlabel('Latent Component 1')
    axs[0, 1].set_ylabel('Latent Component 2')
    axs[0, 1].legend(*scatter_train_times.legend_elements(), title="Time")

    # Plot latent_val_first vs events_val_first
    scatter_val_events = axs[1, 0].scatter(latent_val_2d[:, 0], latent_val_2d[:, 1], c=events_val_first, cmap='coolwarm', alpha=0.7)
    axs[1, 0].set_title('Latent Space vs Events - Validation Data (First Latent)')
    axs[1, 0].set_xlabel('Latent Component 1')
    axs[1, 0].set_ylabel('Latent Component 2')
    axs[1, 0].legend(*scatter_val_events.legend_elements(), title="Events")

    # Plot latent_val_first vs times_val_first
    scatter_val_times = axs[1, 1].scatter(latent_val_2d[:, 0], latent_val_2d[:, 1], c=times_val_first, cmap='coolwarm', alpha=0.7)
    axs[1, 1].set_title('Latent Space vs Times - Validation Data (First Latent)')
    axs[1, 1].set_xlabel('Latent Component 1')
    axs[1, 1].set_ylabel('Latent Component 2')
    axs[1, 1].legend(*scatter_val_times.legend_elements(), title="Time")

    # Show colorbar for event/time labels
    fig.colorbar(scatter_train_events, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1, label='Event/Time')

    # Create the directory if it doesn't exist (if you want to ensure the directory exists)
    save_dir = os.path.dirname(save_directory)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot to the specified directory (including filename)
    plt.tight_layout()
    plt.savefig(save_directory)
    print(f"Plot saved at {save_directory}")

    plt.show()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

def plot_individual_losses(training_stats, filename="training_validation_individual_losses.png"):
    epochs = range(len(training_stats['loss_tr']))
    #training_stats = {key: values[2:] for key, values in training_stats.items()}
    # Create a 4x2 grid of subplots, one for each loss type and dataset (training or validation)
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle("Training and Validation Losses", fontsize=16)

    # Loss plot
    axs[0, 0].plot(epochs, training_stats['loss_tr'], color='blue')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(epochs, training_stats['loss_va'], color='orange')
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss")

    # KL Divergence Loss plot
    axs[1, 0].plot(epochs, training_stats['kl_tr'], color='blue')
    axs[1, 0].set_title("Training KL Divergence Loss")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("KL Loss")

    axs[1, 1].plot(epochs, training_stats['kl_va'], color='orange')
    axs[1, 1].set_title("Validation KL Divergence Loss")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("KL Loss")

    # Image Reconstruction Loss plot
    axs[2, 0].plot(epochs, training_stats['ll_img_tr'], color='blue')
    axs[2, 0].set_title("Training Image Reconstruction Loss")
    axs[2, 0].set_xlabel("Epochs")
    axs[2, 0].set_ylabel("Image Loss")

    axs[2, 1].plot(epochs, training_stats['ll_img_va'], color='orange')
    axs[2, 1].set_title("Validation Image Reconstruction Loss")
    axs[2, 1].set_xlabel("Epochs")
    axs[2, 1].set_ylabel("Image Loss")

    # Time Loss plot
    axs[3, 0].plot(epochs, training_stats['ll_time_tr'], color='blue')
    axs[3, 0].set_title("Training Time Loss")
    axs[3, 0].set_xlabel("Epochs")
    axs[3, 0].set_ylabel("Time Loss")

    axs[3, 1].plot(epochs, training_stats['ll_time_va'], color='orange')
    axs[3, 1].set_title("Validation Time Loss")
    axs[3, 1].set_xlabel("Epochs")
    axs[3, 1].set_ylabel("Time Loss")

    # Adjust layout to avoid overlap and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the suptitle
    plt.savefig(filename, dpi=300)  # Save figure with specified filename and dpi
    plt.show()

# Usage example:
# plot_individual_losses(training_stats)


import numpy as np
import matplotlib.pyplot as plt

def plot_ci_ibs(training_stats, filename="ci_ibs_plot_0.png"):
    epochs = range(len(training_stats['ci_va']))

    # Extract the median values from each list of ci_va, ibs_va, ci_tr, and ibs_tr
    ci_va_median = [np.sort(ci)[1] for ci in training_stats['ci_va']]
    ibs_va_median = [np.sort(ibs)[1] for ibs in training_stats['ibs_va']]
    ci_tr_median = [np.sort(ci)[1] for ci in training_stats['ci_tr']]
    ibs_tr_median = [np.sort(ibs)[1] for ibs in training_stats['ibs_tr']]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot C-Index values (ci_va and ci_tr) on the first subplot (ax1)
    ax1.plot(epochs, ci_va_median, label='Validation C-Index (Median)', color='green')
    ax1.plot(epochs, ci_tr_median, label='Training C-Index (Median)', color='blue')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("C-Index Value")
    ax1.set_title("C-Index: Training vs Validation")
    ax1.legend()

    # Plot IBS values (ibs_va and ibs_tr) on the second subplot (ax2)
    ax2.plot(epochs, ibs_va_median, label='Validation IBS (Median)', color='purple')
    ax2.plot(epochs, ibs_tr_median, label='Training IBS (Median)', color='orange')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("IBS Value")
    ax2.set_title("IBS: Training vs Validation")
    ax2.legend()

    # Adjust layout to make sure there's no overlap
    plt.tight_layout()

    # Save the figure with the specified filename and dpi
    plt.savefig(filename, dpi=300)
    plt.show()

# Usage example:
# plot_training_validation_losses(training_stats, "training_validation_losses.png")
# plot_ci_ibs(training_stats, "ci_ibs_plot.png")

