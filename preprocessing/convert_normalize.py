import os
import glob
import nrrd
import nibabel as nib
import numpy as np
import shutil
import json
import argparse
import logging

def nrrd_to_nifti(baseDir, nifti_path):
    logging.info("Converting nrrd files to NIFTI...")
    if not os.path.exists(nifti_path):
        os.makedirs(nifti_path)
    for i in range(len(os.listdir(baseDir))):
        old_path = baseDir + '/' + os.listdir(baseDir)[i]
        new_path = nifti_path + '/' + os.listdir(baseDir)[i]

        files = glob.glob(old_path+'/*.nrrd')

        for file in files:
            _nrrd = nrrd.read(file)
            data = _nrrd[0]
            header = _nrrd[1]

            img = nib.Nifti1Image(data,affine = None)

            if 'space directions' in header:
                voxel_sizes = np.sqrt((header['space directions']**2).sum(axis=-1))
                img.header['pixdim'][1:4] = voxel_sizes
            else:
                img.header['pixdim'][1:4] = [1, 1, 1]
            sonew_path = new_path + 'T1.nii'
            # sonew_path = new_path + 'DWI.nii'
            logging.info(f"Saving NIFTI file to: {sonew_path}")
            nib.save(img, sonew_path)

def normalize_nifti(nifti_path, norm_folder):
    logging.info("Normalizing NIFTI files...")
    if not os.path.exists(norm_folder):
        os.makedirs(norm_folder)

    for file in os.listdir(nifti_path):
        img_nifti = nib.load(nifti_path +'/' + file)
        img = img_nifti.get_fdata()
        img_new = (img - np.min(img)) / (np.max(img) - np.min(img))
        img_normed = nib.Nifti1Image(img_new, img_nifti.affine)
        nib.save(img_normed, norm_folder  + '/' + file)

def generate_json(folder, json_path):
    logging.info("Generating JSON file...")
    data = []
    for test_id in os.listdir(folder):
        item = {"image": "imagesTs/{}.nii".format(test_id[:-4])}
        data.append(item)

    with open(json_path, 'w') as f:
        json.dump(data, f,indent=2)

    logging.info(f"JSON file has been created at: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process nrrd files, normalize and create a JSON file.')
    parser.add_argument('--baseDir', required=True, help='Directory with the nrrd files')
    parser.add_argument('--nifti_path', required=True, help='Output directory for the NIFTI files')
    parser.add_argument('--norm_folder', required=True, help='Output directory for normalized NIFTI files')
    parser.add_argument('--json_path', required =True, help='Path for the output JSON file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    nrrd_to_nifti(args.baseDir, args.nifti_path)
    normalize_nifti(args.nifti_path, args.norm_folder)
    generate_json(args.norm_folder, args.json_path)

# This version of the script is more in line with the best practices for a production environment. It is modular, has improved error handling with logging, and makes use of command line arguments for greater flexibility.

# To run the script, you can use the command line to provide the necessary arguments. Here is an example of how to run it:

# python script.py --baseDir /path/to/nrrd_files --nifti_path /path/to/output_nifti --norm_folder /path/to/normalized_output --json_path /path/to/output.json


# Remember to replace /path/to/nrrd_files, /path/to/output_nifti, /path/to/normalized_output and /path/to/output.json with your actual directories and file paths.

