import cv2
import os
import numpy as np
from descriptor import glcm, bitdesc
 
# List of descriptors
descriptors = [glcm, bitdesc]
 
def process_datasets(root_folder, descriptors):
    all_features_glcm = []  # List to store features for GLCM
    all_features_bitdesc = []  # List to store features for BITdesc
 
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                file_name = f'{relative_path.split(os.sep)[0]}_{file}'  
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
 
                print(f"Processing file: {image_rel_path}")
 
                img = cv2.imread(image_rel_path, 0)
                if img is not None:
                    # Extract features using GLCM
                    features_glcm = descriptors[0](img)
                    if features_glcm is not None:
                        features_glcm = features_glcm + [folder_name, relative_path]
                        all_features_glcm.append(features_glcm)
 
                    # Extract features using BITdesc
                    features_bitdesc = descriptors[1](img)
                    if features_bitdesc is not None:
                        features_bitdesc = features_bitdesc + [folder_name, relative_path]
                        all_features_bitdesc.append(features_bitdesc)
                else:
                    print(f"Failed to read image: {image_rel_path}")
 
    # Convert lists to numpy arrays and save
    signatures_glcm = np.array(all_features_glcm)
    np.save('signatures_glcm.npy', signatures_glcm)
 
    signatures_bitdesc = np.array(all_features_bitdesc)
    np.save('signatures_bitdesc.npy', signatures_bitdesc)
 
    print('Successfully stored!')
 
# Process datasets
process_datasets('./images', descriptors)