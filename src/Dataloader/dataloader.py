import os
import dicom2nifti
import numpy as np
import nibabel as nib
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd, ScaleIntensityRanged, Resized, ToTensord
from monai.utils import set_determinism
import torch
import os
import pydicom
import numpy as np
import pandas as pd
import dicom2nifti  # Ensure you have this installed
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd, ScaleIntensityRanged, Resized, ToTensord
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
import torch
import shutil 
# Set deterministic training behavior
set_determinism(seed=0)
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, Orientationd, Spacingd, ScaleIntensityRanged, Resized, ToTensord, RandRotate90d, RandZoomd, RandFlipd


import os
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd, ScaleIntensityRanged, Resized, ToTensord
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
import torch


# Set deterministic training behavior
set_determinism(seed=0)


class CustomDataset(Dataset):
    def __init__(self, root_dir, excel_path, transform=None):
        self.root_dir = root_dir
        self.data_frame = pd.read_excel(excel_path)
        
        # Collect paths to each NIfTI file
        self.nifti_files = [os.path.join(root_dir, name, 'output.nii.gz') for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        self.transform = transform


    def __getitem__(self, idx):
        nifti_file = self.nifti_files[idx]
        patient_id = os.path.basename(os.path.dirname(nifti_file))  # Get the patient folder name
        row = self.data_frame[self.data_frame['Patient-ID'] == patient_id]


        # Prepare dictionary for MONAI transformations
        image_dict = {"vol": nifti_file}


        # Apply transforms
        if self.transform:
            image_dict = self.transform(image_dict)


        # Extract the transformed image tensor
        image = image_dict["vol"]


        # Retrieve survival time and event status
        survival_time = row['overall_survival_months'].values[0]
        event = row['vital_status'].values[0]  # Assuming 'vital_status' is the event indicator
        
        return image, survival_time, event


    def __len__(self):
        return len(self.nifti_files)


def get_data_loaders(root_dir, excel_path, batch_size=3, test_size=0.25):
    """Prepare the train and test data loaders with stratification."""
    
    # Define transformations
    transforms = Compose([
        LoadImaged(keys=["vol"]),
        EnsureChannelFirstD(keys=["vol"]),
        Orientationd(keys=["vol"], axcodes='RAS'),
        Spacingd(keys=["vol"], pixdim=(1.5, 1.5, 2.0), mode='trilinear'),
        ScaleIntensityRanged(keys=["vol"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["vol"], spatial_size=(128, 128, 64)),
        # RandRotate90d(keys=["vol"], prob=0.5, max_k=3),  # Random rotations
        # RandZoomd(keys=["vol"], prob=0.5, min_zoom=0.8, max_zoom=1.2),  # Random zoom
        # RandFlipd(keys=["vol"], prob=0.5, spatial_axis=[0, 1]),  # Random flipping in two axes
        ToTensord(keys=["vol"])
    ])


    dataset = CustomDataset(root_dir, excel_path, transform=transforms)


    # Stratify by event labels
    labels = [dataset.data_frame[dataset.data_frame['Patient-ID'] == os.path.basename(os.path.dirname(nifti))]['vital_status'].values[0]
              for nifti in dataset.nifti_files]
    
    train_indices, test_indices = train_test_split(
        range(len(labels)),
        test_size=test_size,
        stratify=labels
    )


    # Create train and test subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)


    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader


def check_for_nans(dataloader):
    """Check for NaN values in batches of data."""
    nan_found = False


    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            print(f"Skipping batch {batch_idx} due to missing data.")
            continue
        
        images, survival_times, events = batch
        
        # Check for NaNs in image tensors
        if torch.isnan(images).any():
            print(f"NaN found in images at batch {batch_idx}")
            nan_found = True


        # Check for NaNs in survival times and events
        if torch.isnan(survival_times).any():
            print(f"NaN found in survival_times at batch {batch_idx}")
            nan_found = True


        if torch.isnan(events).any():
            print(f"NaN found in events at batch {batch_idx}")
            nan_found = True


    if not nan_found:
        print("No NaNs found in the DataLoader.")

import matplotlib.pyplot as plt
import random


def show_random_slices(dataloader, num_samples=3):
    """Display random slices from a DataLoader."""
    for i in range(num_samples):
        # Get a random batch
        images, survival_times, events = next(iter(dataloader))


        # Select a random sample from the batch
        idx = random.randint(0, images.shape[0] - 1)
        image = images[idx].squeeze().numpy()  # Convert to numpy array and remove channel dimension if present
        survival_time = survival_times[idx].item()
        event = events[idx].item()


        # Display the middle slice in the z-axis
        middle_slice_idx = image.shape[0] // 2
        middle_slice = image[middle_slice_idx, :, :]


        # Plot the slice
        plt.figure(figsize=(6, 6))
        plt.imshow(middle_slice, cmap='gray')
        plt.title(f"Survival Time: {survival_time}, Event: {event}")
        plt.axis('off')
        plt.show()

# Example usage
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(
        root_dir='Converted_NIfTI',
        excel_path='input/crlm.xlsx',
        batch_size=3,
        test_size=0.30 
    )
    print("len(train_loader):", len(train_loader))
    print("len(test_loader):", len(test_loader))


    # print("Checking train_loader for NaNs:")
    # check_for_nans(train_loader)


    # print("\nChecking test_loader for NaNs:")
    # check_for_nans(test_loader)



    # # Example usage
    # print("Displaying random slices from train_loader:")
    # show_random_slices(train_loader)


    # print("\nDisplaying random slices from test_loader:")
    # show_random_slices(test_loader)

