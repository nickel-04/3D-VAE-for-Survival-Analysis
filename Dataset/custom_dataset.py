import os
import pydicom
import numpy as np
import nibabel as nib
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd, ScaleIntensityRanged, Resized, ToTensord
from monai.utils import set_determinism

# Set deterministic training behavior
set_determinism(seed=0)

class DicomToNiftiConverter:
    def __init__(self, source_folder, target_folder, target_filename='output.nii.gz'):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.target_filename = target_filename

    def convert(self):
        """Converts a series of DICOM images within a folder into a single compressed NIfTI file."""
        nifti_path = os.path.join(self.target_folder, self.target_filename)
        
        # Check if NIfTI file already exists
        if os.path.exists(nifti_path):
            print(f"NIfTI file already exists at {nifti_path}. Skipping conversion.")
            return nifti_path

        dicom_files = [os.path.join(self.source_folder, f) for f in os.listdir(self.source_folder) if f.endswith('.dcm')]
        if not dicom_files:
            print(f"No DICOM files found in {self.source_folder}.")
            return None

        try:
            dicom_images = [pydicom.dcmread(file_path) for file_path in dicom_files]
            dicom_images.sort(key=lambda x: int(x.InstanceNumber))
            image_data = np.stack([img.pixel_array for img in dicom_images])
        except Exception as e:
            print(f"Failed to read DICOM files: {e}")
            return None

        try:
            affine = np.diag([1, 1, 1, 1])  # Placeholder affine matrix
            nifti_image = nib.Nifti1Image(image_data, affine=affine)
            os.makedirs(self.target_folder, exist_ok=True)
            nib.save(nifti_image, nifti_path)
            print(f"NIfTI file saved at {nifti_path}")
            return nifti_path
        except Exception as e:
            print(f"Failed to convert to NIfTI: {e}")
            return None


def load_images(root_path):
    """Navigate through folder levels to find and convert DICOM files to NIfTI."""
    for root, _, _ in os.walk(root_path):
        dicom_files = [f for f in os.listdir(root) if f.endswith('.dcm')]
        if len(dicom_files) > 1:
            nifti_folder = os.path.join(root_path, 'nifti')
            nifti_file = os.path.join(nifti_folder, 'output.nii.gz')
            
            # Check if the NIfTI file already exists
            if os.path.exists(nifti_file):
                print(f"NIfTI file already exists at {nifti_file}. Skipping conversion.")
                return nifti_file
            
            os.makedirs(nifti_folder, exist_ok=True)
            converter = DicomToNiftiConverter(root, nifti_folder)
            return converter.convert()
    return None


class CustomDataset(Dataset):
    def __init__(self, root_dir, excel_path, transform=None):
        self.root_dir = root_dir
        self.data_frame = pd.read_excel(excel_path)
        self.patient_folders = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        self.transform = transform

    def __getitem__(self, idx):
        folder_name = self.patient_folders[idx]
        patient_id = os.path.basename(folder_name)
        row = self.data_frame[self.data_frame['Patient-ID'] == patient_id]
        image_path = load_images(folder_name)

        if image_path is None:
            return None  # Handle according to your application's needs

        # Prepare a dictionary for the transform
        image_dict = {"vol": image_path}

        # Apply transforms
        if self.transform:
            image_dict = self.transform(image_dict)  # Pass the dictionary containing the file path

        # The transformed dictionary should now have "vol" as a tensor
        image = image_dict["vol"]  # This is now the tensor directly

        # Return the image and other data
        survival_time = row['overall_survival_months'].values[0]
        vital_status = row['vital_status'].values[0]
        
        return image, survival_time, vital_status

    def __len__(self):
        return len(self.patient_folders)


def get_data_loaders(root_dir, excel_path, batch_size=3, test_size=0.25):
    """Prepare the train and test data loaders with stratification."""
    from sklearn.model_selection import train_test_split
    import torch

    # Define transformations
    transforms = Compose([
        LoadImaged(keys=["vol"]),  # Use the correct key in your dataset
        EnsureChannelFirstD(keys=["vol"]),
        Spacingd(keys=["vol"], pixdim=(1.0, 1.0, 1.0), mode='trilinear'),
        Orientationd(keys=["vol"], axcodes='RAS'),
        ScaleIntensityRanged(keys=["vol"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["vol"], spatial_size=(128, 128, 64)),
        ToTensord(keys=["vol"])
    ])

    dataset = CustomDataset(root_dir, excel_path, transform=transforms)

    # Split indices into train/test
    labels = [dataset.data_frame[dataset.data_frame['Patient-ID'] == os.path.basename(p)]['vital_status'].values[0]
              for p in dataset.patient_folders]
    
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

import torch

def check_for_nans(dataloader):
    """Check for NaN values in batches of data."""
    nan_found = False

    for batch_idx, (images, survival_times, vital_statuses) in enumerate(dataloader):
        # Check for NaNs in image tensors
        if torch.isnan(images).any():
            print(f"NaN found in images at batch {batch_idx}")
            nan_found = True

        # Check for NaNs in survival times and vital statuses
        if torch.isnan(torch.tensor(survival_times)).any():
            print(f"NaN found in survival_times at batch {batch_idx}")
            nan_found = True

        if torch.isnan(torch.tensor(vital_statuses)).any():
            print(f"NaN found in vital_statuses at batch {batch_idx}")
            nan_found = True

    if not nan_found:
        print("No NaNs found in the DataLoader.")

# # Example usage
# if __name__ == '__main__':
#     train_loader, test_loader = get_data_loaders(
#         root_dir='NBIA-Download',
#         excel_path='Dataset/crlm.xlsx',
#         batch_size=2,
#         test_size=0.25
#     )


#from Data_loader import get_data_loaders

# Example usage
# train_loader, test_loader = get_data_loaders(
#     root_dir='/home/nikhil/Downloads/tcia_download/NBIA-Download',
#     excel_path='/home/nikhil/Downloads/tcia_download/Dataset/crlm.xlsx',
#     batch_size=2,
#     test_size=0.25
# )

# # Now you can use train_loader and test_loader in your model
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(
    root_dir='NBIA-Download',
    excel_path='Dataset/crlm.xlsx',
    batch_size=2,
    test_size=0.25 
    )
    print("len(train_loader)",len(train_loader))
    print("len(test_loader)",len(test_loader))

    print("Checking train_loader for NaNs:")
    check_for_nans(train_loader)

    print("\nChecking test_loader for NaNs:")
    check_for_nans(test_loader)
