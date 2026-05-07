#main

import ants
import numpy as np
import matplotlib.pyplot as plt
from functions.loadnorm import loadnorm
from convertpng import convertpng
from hudict import HU_dict
from functions.cannyfxn import canny_process, canny_feature
from functions.logfxn import log_process, log_feature
from functions.watershedfxn import watershed_process, watershed_feature
import os
import pydicom
from skimage.segmentation import watershed
from scipy import ndimage as ndi

convertpng('imgdata/fixa.png')
convertpng('imgdata/mova.png')

fi = ants.image_read("imgdata/fixa.nii.gz")
mi = ants.image_read("imgdata/mova.nii.gz")

#File hudict.py has the Hounsfield unit boundaries for many common tissue/matter types. 
upper = HU_dict["soft_tissue"][0]
lower = HU_dict["soft_tissue"][1]


#load in data and complete initial rigid registration

[fi, mi, initx] = loadnorm("imgdata/fixa.nii.gz","imgdata/mova.nii.gz", lower, upper)
print('loaded and registered images')

#laplacian of gaussian (soft tissue only)

[testlog,testtx1] = log_process(fi,mi,0.7)
ants.image_write(testlog, 'testlog.nii.gz')
print('laplacian of gaussian complete')

#canny (soft tissue only)

[testcan,testtx2] = canny_process(fi,mi)
ants.image_write(testcan, 'testcanny.nii.gz')
print('canny complete')

#watershed (soft tissue only)

[testwtr,testtx3] = watershed_process(fi,mi)
ants.image_write(testwtr, 'testwtr.nii.gz')
print('watershed complete')


#BMFA registration 

#log (BMFA)

bone_u = HU_dict["bone_gen"][0]
bone_l = HU_dict["bone_gen"][1]

fat_u = HU_dict["fat"][0]
fat_l = HU_dict["fat"][1]

fst_u = HU_dict["air"][0]
fst_l = HU_dict["air"][1]


midirect = "imgdata/mova.nii.gz"
fidirect = "imgdata/fixa.nii.gz"

#Water Dixon image uses Bone
[bfi, bmi, binitx] = loadnorm(fidirect,midirect, bone_l, bone_u)

#Fat Dixon image uses Fat
[ffi, fmi, finitx] = loadnorm(fidirect,midirect, fat_l, fat_u)

#Out of Phase Dixon image uses Water and Fat (water is 0 so just using fat again)
[opfi, opmi, opinitx] = loadnorm(fidirect,midirect, fat_l, fat_u)

#In Phase Dixon image uses Fat and Soft Tissue
[ipfi, ipmi, ipinitx] = loadnorm(fidirect,midirect, fst_l, fst_u)


#Make feature maps

[bffeat, bmfeat] = log_feature(bfi, bmi, 0.07)
[fffeat, fmfeat] = log_feature(ffi, fmi, 0.07)
[opffeat, opmfeat] = log_feature(opfi, opmi, 0.07)
[ipffeat, ipmfeat] = log_feature(ipfi, ipmi, 0.07)

# Define the multi-metric setup
# Format: ( (name, fixed, moving, weight, sampling), ... )
extra_metrics = [
    ("MeanSquares",  bffeat, bmfeat, 0.25),
    ("MeanSquares",  fffeat, fmfeat, 0.25),
    ("MeanSquares",  opffeat, opmfeat, 0.25),
    ("MeanSquares",  ipffeat, ipmfeat, 0.25)
]

# Run registration
reg = ants.registration(
    fixed=fi,
    moving=mi,
    type_of_transform='SyN',
    multivariate_extras=extra_metrics
    )



#canny (BMFA)

#watershed (BMFA)

#ssim/sift/jacobian

#plotting