
import ants
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def canny_process(fi, mi):
    # Create feature map
    fixed_feature = ants.iMath(fi, "Canny",1, 5, 12)
    moving_feature = ants.iMath(mi, "Canny",1, 5, 12)

    #Deformable registration (SyN) using multiparameter feature maps
    mytx = ants.registration(
        fixed=fi, 
        moving=mi, 
        type_of_transform='SyN',
        multivariate_extras=[('MeanSquares', fixed_feature, moving_feature, 0.5, 0.0)]
    )
    print('registered images')
    registered_img = ants.apply_transforms(fixed=fi, moving=mi,
                                      transformlist=mytx['fwdtransforms'])

    return(registered_img,mytx)


def canny_feature(fi,mi):
    fix = ants.iMath(fi, "Canny",1, 5, 12)
    mix = ants.iMath(mi, "Canny",1, 5, 12)
    return(fix,mix)

# [testlog,testtx] = canny_process(fi,mi)
# ants.image_write(testlog, 'testcanny.nii.gz')


