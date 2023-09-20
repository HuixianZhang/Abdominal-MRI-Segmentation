import pandas as pd
import shutil
import os

old_path = 'Y:/CCHMC/MRI2/QC.T1/'
count = 0
folders_without_required_files = []

for folder in os.listdir(old_path):
    print(folder)
    has_required_file = False  # This flag will help us track if the current folder has a file that meets our requirements
    for file in os.listdir(old_path + folder):
        if ('_W' in file and 'FAT' not in file and 'InPhase' not in file and 'OutPhase' not in file) or ('LAVA' in file and 'FAT' not in file and 'InPhase' not in file and 'OutPhase' not in file):
            count += 1
            has_required_file = True  # Set the flag to True if we find a file that meets our requirements

    if not has_required_file:
        folders_without_required_files.append(folder)  # If no file met the requirement in the folder, add the folder to our list

print("Number of matching files:", count)
print("Folders without required files:", folders_without_required_files)
