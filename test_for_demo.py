import argparse
import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, mode='nearest',prefilter=False)
    return img_resampled


def segmentation(filename):
    
    output_directory = "outputs/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pth = 'runs/test/model.pt'
    model = SwinUNETR(
        img_size=64,
        in_channels=1,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    )
    model_dict = torch.load(model_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    
    with torch.no_grad():
        test_input = nib.load(filename).get_fdata()

         # Get the shape of the input image 
        h, w, d = test_input.shape
        target_shape = (h, w, d)
        
        # Normalize the input image
        test_input = (test_input - np.min(test_input)) / (np.max(test_input) -np.min(test_input))
       
        if d < 64:
            #  0 is nearest, and 1 is bilinear
            test_input = resample_3d(test_input, (h,w,64))

        test_input_tensor = torch.from_numpy(test_input).unsqueeze(0).unsqueeze(0).float().to(device)
        # # (64,64,64) is the image size, and 4 is the sliding window batch size,
        # # overlap is the overlap rate of sliding window, higher overlap will get better performance
        sliding_window_output = sliding_window_inference(
            test_input_tensor, (64, 64, 64), 4, model, overlap=0.5, mode="gaussian"
        )
        test_output = torch.softmax(sliding_window_output, 1).cpu().numpy()
        test_output = np.argmax(test_output, axis=1).astype(np.uint8)[0]
        
        test_output = resample_3d(test_output, target_shape)
        # return the output mask, a numpy array
        return test_output
    

if __name__ == "__main__":
    segmentation(filename= '0001/T2.nii')