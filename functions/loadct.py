

import ants
import cv2
import matplotlib.pyplot as plt
from functions.edgemap import make_edgemap
from functions.multiframed import create_multiframe_dicom
import ants.utils
import scipy
import cv2
import dicom2nifti
import os
import pydicom
import numpy as np


def loadct(input_dir, output_file):
    # Read all DICOM files in the directory
    slices = []
    for f in os.listdir(input_dir):
        if f.endswith(".dcm"):
            slices.append(pydicom.dcmread(os.path.join(input_dir, f)))

    # Sort slices based on Instance Number (or Image Position Patient)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    ds = slices[0]

    # Extract and stack pixel data
    pixel_arrays = [s.pixel_array for s in slices]
    stacked_pixels = np.stack(pixel_arrays, axis=0)

    # Convert back to bytes for the PixelData attribute
    ds.PixelData = stacked_pixels.tobytes()

    # Update multi-frame specific tags
    ds.NumberOfFrames = len(slices)
    ds.save_as(output_file, write_like_original=False)

    return()


def normct(input_dir, lower, upper):
    #normalizing CT intensities to be read on the same scale as MRI
    mi = ants.image_read(input_dir)

    # Convert to numpy for intensity manipulation
    img_arr = mi.numpy()

    # Standard CT normalization: Clip to window (e.g., Soft Tissue window -150 to 350 HU)
    # Or standard wide window (-1000 to 1000 HU)
    img_arr = img_arr.clip(lower, upper)

    img_min = img_arr.min()
    img_max = img_arr.max()
    norm_arr = (img_arr - img_min) / (img_max - img_min)

    # Convert back to ANTs image (preserving original header/spacing)
    norm_img = mi.new_image_like(norm_arr)

    # # N4 Bias Field Correction (often used even in CT to smooth intensity gradients)
    # corrected_img = ants.n4_bias_field_correction(norm_img)
    return(norm_img)


def regsct(fi, mi):
    #initial rigid registration of sCT to fixed image after normalization

    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Rigid' )
    mi2 = ants.apply_transforms(fixed=fi, moving=mi,
                                      transformlist=mytx['fwdtransforms'])

    return(fi, mi2, mytx)

from matplotlib import pyplot as plt

#ref: https://antspy.readthedocs.io/en/latest/registration.html

from matplotlib import pyplot as plt
# print(dir(ants.utils))

fidirect = 'Data/ANON50819/DOE^JOHN_ANON50819_MR_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_sCT-Dixon.opp_n192__00000'
finside = 'Data/ANON50819/DOE^JOHN_ANON50819_MR_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_sCT-Dixon.opp_n192__00000/2.16.840.1.114362.1.12209795.24357611168.727506466.555.202.dcm'
midirect = 'Data/ANON50819/DOE^JOHN_ANON50819_CT_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_SyntheticCT.HU_n192__00000/'

test = loadct(midirect,'output/sctnifti.nii.gz')
test2 = normct('output/sctnifti.nii.gz',-100,350)

ants.image_write(test2, 'fxntest.nii.gz')


dicom_file = finside

m2 = ants.dicom_read(midirect)



# Directory containing single-slice DICOM files
input_dir = midirect
output_file = 'output/sctnifti.dcm'

# Read all DICOM files in the directory
slices = []
for f in os.listdir(input_dir):
    if f.endswith(".dcm"):
        slices.append(pydicom.dcmread(os.path.join(input_dir, f)))

# Sort slices based on Instance Number (or Image Position Patient)
slices.sort(key=lambda x: int(x.InstanceNumber))
ds = slices[0]

# Extract and stack pixel data
pixel_arrays = [s.pixel_array for s in slices]
stacked_pixels = np.stack(pixel_arrays, axis=0)

# Convert back to bytes for the PixelData attribute
ds.PixelData = stacked_pixels.tobytes()

# Update multi-frame specific tags
ds.NumberOfFrames = len(slices)
ds.save_as(output_file, write_like_original=False)


meta_data = ants.read_image_metadata(dicom_file)

fi = ants.image_read(dicom_file)
mi = ants.image_read(output_file)

# Convert to numpy for intensity manipulation
img_arr = mi.numpy()

# Standard CT normalization: Clip to window (e.g., Soft Tissue window -150 to 350 HU)
# Or standard wide window (-1000 to 1000 HU)
img_arr = img_arr.clip(-150, 350)

img_min = img_arr.min()
img_max = img_arr.max()
norm_arr = (img_arr - img_min) / (img_max - img_min)

# Convert back to ANTs image (preserving original header/spacing)
norm_img = mi.new_image_like(norm_arr)

# # N4 Bias Field Correction (often used even in CT to smooth intensity gradients)
# corrected_img = ants.n4_bias_field_correction(norm_img)

ants.image_write(norm_img, 'normalized_ct.nii.gz')

mytx = ants.registration(fixed=fi, moving=norm_img, type_of_transform = 'SyN' )
norm_img2 = ants.apply_transforms(fixed=fi, moving=norm_img,
                                      transformlist=mytx['fwdtransforms'])

ants.image_write(norm_img2, 'output/normimg2.nii.gz')
ants.image_write(fi, 'output/fi2.nii.gz')