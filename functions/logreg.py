

import ants
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.edgemap import make_edgemap
from functions.multiframed import create_multiframe_dicom
import cv2
import os
import pydicom

from matplotlib import pyplot as plt

#ref: https://antspy.readthedocs.io/en/latest/registration.html

from matplotlib import pyplot as plt
# print(dir(ants.utils))

fidirect = 'Data/ANON50819/DOE^JOHN_ANON50819_MR_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_sCT-Dixon.opp_n192__00000'
finside = 'Data/ANON50819/DOE^JOHN_ANON50819_MR_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_sCT-Dixon.opp_n192__00000/2.16.840.1.114362.1.12209795.24357611168.727506466.555.202.dcm'
midirect = 'Data/ANON50819/DOE^JOHN_ANON50819_CT_2023-05-10_143745_RADIATION.ONCOLOGY.MR.TREATMENT.PLAN_SyntheticCT.HU_n192__00000/'



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

mi = norm_img

ants.image_write(fi, 'output/fi.nii.gz')
ants.image_write(mi, 'output/mi.nii.gz')
log_img = fi.iMath("Laplacian",0.5)

# Create a feature (e.g., a mask or edge image)

fixed_feature = ants.iMath(fi, "Laplacian",0.7)
moving_feature = ants.iMath(mi, "Laplacian",0.7)

mytx = ants.registration(
    fixed=fi, 
    moving=mi, 
    type_of_transform='SyN',
    multivariate_extras=[('MeanSquares', fixed_feature, moving_feature, 0.5, 0.0)]
)



newmove = mi



registered_img2 = ants.apply_transforms(fixed=fi, moving=mi,
                                      transformlist=mytx['fwdtransforms'])

ants.image_write(fixed_feature, 'output/logfi.nii.gz')
ants.image_write(registered_img2, 'output/regimgfwd2.nii.gz')



#extracting deformation matrix
deformation_field = mytx['fwdtransforms'][0]
transform_params = mytx['fwdtransforms']

#applying to other image...
# transformedimg = ants.applytransforms(fixed=fia,moving=mia,transformlist=transformparams)

#getting jacobian as well
jac = ants.create_jacobian_determinant_image(fi,mytx['fwdtransforms'][0],do_log=False)


