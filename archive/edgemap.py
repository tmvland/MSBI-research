
#Generate edge map

import nibabel as nib
import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_edgemap(mri_slice):

    # img = nib.load('mri_volume.nii.gz')
    # data = img.get_fdata()

    # slice_idx = data.shape[2] // 2
    # mri_slice = data[:, :, slice_idx]

    # mri_slice = cv2.imread('fixa.png')

    # Normalize the data to 0-255 for OpenCV
    mri_slice_norm = cv2.normalize(mri_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gaussian blur 
    blurred = cv2.GaussianBlur(mri_slice_norm, (3, 3), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    #Laplacian of Gaussian edge detection (likely will want this one)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    return(edges,log,mri_slice)

#testing functionality
# [edges,log,mri_slice] = make_edgemap()

# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(mri_slice, cmap='gray'), plt.title('Original Slice')
# plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Map')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(mri_slice, cmap='gray'), plt.title('Original Slice')
# plt.subplot(122), plt.imshow(log, cmap='gray'), plt.title('Edge Map')
# plt.show()
