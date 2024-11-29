import os
import dicom2nifti


# Define source and target directories
source_root = 'NBIA-Download'
target_root = 'Converted_NIfTI'


# Create the target root directory if it doesn't exist
os.makedirs(target_root, exist_ok=True)


# Loop over each folder in the NBIA-Download directory
for folder in sorted(os.listdir(source_root)):
    source_folder = os.path.join(source_root, folder)
    
    # Skip if not a directory
    if not os.path.isdir(source_folder):
        continue


    # Enter the inner folder
    inner_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    if len(inner_folders) != 1:
        print(f"Unexpected folder structure in {source_folder}")
        continue


    # Enter the second level folder
    second_level_folder = os.path.join(source_folder, inner_folders[0])
    subfolders = [f for f in os.listdir(second_level_folder) if os.path.isdir(os.path.join(second_level_folder, f))]


    # Identify the folder with multiple DICOM files
    target_dicom_folder = None
    for subfolder in subfolders:
        subfolder_path = os.path.join(second_level_folder, subfolder)
        dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        
        # Select folder with more than one DICOM file
        if len(dicom_files) > 1:
            target_dicom_folder = subfolder_path
            break


    # If a target DICOM folder was found, proceed with conversion
    if target_dicom_folder:
        target_folder = os.path.join(target_root, folder)
        os.makedirs(target_folder, exist_ok=True)
        
        # Convert DICOM to NIfTI
        try:
            dicom2nifti.convert_directory(target_dicom_folder, target_folder, compression=True, reorient=True)
            
            # Rename the converted file to "output.nii.gz"
            nifti_files = [f for f in os.listdir(target_folder) if f.endswith('.nii.gz')]
            if nifti_files:
                original_nifti_path = os.path.join(target_folder, nifti_files[0])
                renamed_nifti_path = os.path.join(target_folder, 'output.nii.gz')
                os.rename(original_nifti_path, renamed_nifti_path)
                print(f"Converted {folder} to NIfTI at {renamed_nifti_path}")
            else:
                print(f"No NIfTI file found after conversion for {folder}")
                
        except Exception as e:
            print(f"Failed to convert {folder} due to error: {e}")
    else:
        print(f"No suitable DICOM folder found for {folder}")



# import os
# import dicom2nifti


# # Define source and target directories
# source_root = 'nap'
# target_root = 'Converted_NIfTI'


# # Create the target root directory if it doesn't exist
# os.makedirs(target_root, exist_ok=True)


# # Loop over each folder in the NBIA-Download directory
# for folder in sorted(os.listdir(source_root)):
#     source_folder = os.path.join(source_root, folder)
    
#     # Skip if not a directory
#     if not os.path.isdir(source_folder):
#         continue


#     # Enter the inner folder
#     inner_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
#     if len(inner_folders) != 1:
#         print(f"Unexpected folder structure in {source_folder}")
#         continue


#     # Enter the second level folder
#     second_level_folder = os.path.join(source_folder, inner_folders[0])
#     subfolders = [f for f in os.listdir(second_level_folder) if os.path.isdir(os.path.join(second_level_folder, f))]


#     # Identify the folder with multiple DICOM files
#     target_dicom_folder = None
#     for subfolder in subfolders:
#         subfolder_path = os.path.join(second_level_folder, subfolder)
#         dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        
#         # Select folder with more than one DICOM file
#         if len(dicom_files) > 1:
#             target_dicom_folder = subfolder_path
#             break


#     # If a target DICOM folder was found, proceed with conversion
#     if target_dicom_folder:
#         target_folder = os.path.join(target_root, folder)
#         os.makedirs(target_folder, exist_ok=True)
#         nifti_file = os.path.join(target_folder, 'output.nii.gz')
        
#         # Convert DICOM to NIfTI
#         try:
#             dicom2nifti.convert_directory(target_dicom_folder, target_folder, compression=True, reorient=True)
#             print(f"Converted {folder} to NIfTI at {nifti_file}")
#         except Exception as e:
#             print(f"Failed to convert {folder} due to error: {e}")
#     else:
#         print(f"No suitable DICOM folder found for {folder}")


