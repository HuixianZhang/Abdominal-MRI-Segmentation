import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_fill_holes, binary_dilation
import os

def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}

    for c in for_which_classes:
        mask = image == c
        lmap, num_objects = label(mask.astype(int))
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                if object_sizes[object_id] != maximum_size:
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image

def fill_holes7(img: np.ndarray, applied_labels: np.ndarray) -> np.ndarray:
    output = np.zeros(img.shape, dtype=int)
    for i in applied_labels:
        tmp = np.zeros(img.shape, dtype=bool)
        binary_dilation(tmp, structure=None, iterations=-1, mask=img != i, origin=0, border_value=1, output=tmp)
        output[np.logical_not(tmp)] = i
    return output

path = 'outputs/test1/'
output_path = 'outputs_UM_T1_postprocessed/'

# Create the directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

for file in os.listdir(path):
    mask_segmentation = nib.load(path + file).get_fdata()
    header = nib.load(path + file).header
    mask_segmentation = np.array(mask_segmentation)
    mask_connected = remove_all_but_the_largest_connected_component(mask_segmentation,[1,2],5,None)
    mask_filledinholes = fill_holes7(mask_connected,[1,2]).astype(int)
    img = nib.Nifti1Image(mask_filledinholes, None, header=header)
    sonew_path = output_path + file
    nib.save(img,sonew_path)
