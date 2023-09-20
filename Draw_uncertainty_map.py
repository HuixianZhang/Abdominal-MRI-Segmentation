# To draw uncertainy map according to the probabilities in model prediction.
# Import necessary libraries
import os
import subprocess
import numpy as np
from scipy import stats
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import scipy.ndimage as ndimage

# Set the number of runs
num_runs = 3

# Define the folder containing the labels
label_folder = 'dataset/dataset0/labelsTs/'

# Initialize lists to store file paths and mask probabilities
file_paths = []
mask_probability = []

# Loop through the number of runs
for i in range(1, num_runs + 1):
    print(f"Run {i}")
    
    # Set the CUDA device to be used
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    
    # Run the test script and save the output to a temporary file
    with open("results_temp.txt", "w") as outfile:
        subprocess.run([
            "python", "test.py",
            "--infer_overlap=0.5",
            "--data_dir=dataset/dataset0/",
            "--pretrained_dir=runs/test/",
            "--pretrained_model_name=model.pt",
            "--json_list=dataset_1.json"
        ], env=env, stdout=outfile)

    # Read the output from the temporary file and store the file paths and mask probabilities
    with open("results_temp.txt", "r") as infile:
        lines = infile.read().splitlines()
        file_paths.append(lines)
        mask_probability.append([np.load(file_path) for file_path in lines])

# Calculate the mean mask probability across all runs
mean_mask_probability = [np.mean([run_masks[i] for run_masks in mask_probability], axis=0) for i in range(len(mask_probability[0]))]

# Define the output directory and create it if it doesn't exist
output_directory = "./mean_probability_outputs_masks"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define a function to resample a 3D image to a target size
def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, mode='nearest', prefilter=False)
    return img_resampled

# Loop through the mean mask probabilities and file paths
for i, (prob, file_path) in enumerate(zip(mean_mask_probability, file_paths[0])):
    liver_probability = prob[1]
    spleen_probability = prob[2]
    label_path = label_folder + file_path.split('/')[-1].split('.')[0] + '.nii'

    # Load the label data
    label = nib.load(label_path).get_fdata()
    
    # Calculate the model uncertainty
    P_forground = liver_probability + spleen_probability
    P_background = 1 - P_forground
    U_ts = -(P_forground * np.log(P_forground) + P_background * np.log(P_background))
    U_ts = np.flip(U_ts, axis=0)
    U_ts = np.rot90(U_ts, k=-1)
    U_ts = resample_3d(U_ts, label.shape)
    num_slices = U_ts.shape[2]
    
    # Save all slices for the current instance
    for slice_idx in range(num_slices):
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(U_ts[:, :, slice_idx], cmap='gray')
        ax.set_title(f'Model Uncertainty - {file_path} - Slice {slice_idx}', {'fontsize': 4})
        ax.axis('off')
        plt.tight_layout()

        # Save the figure using the file path as the instance name and include the slice index
        instance_name = os.path.splitext(os.path.basename(file_path))[0]
        plt.savefig(f'./figures/{instance_name}_uncertainty_slice_{slice_idx}.png')
        plt.close(fig)  # Close the figure to free up memory
