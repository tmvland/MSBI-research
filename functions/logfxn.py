import ants
import numpy as np
import matplotlib.pyplot as plt
from functions.loadct import loadct, normct, regsct
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def log_process(fi, mi,alpha):
    # Create a feature (e.g., a mask or edge image)
    fixed_feature = ants.iMath(fi, "Laplacian",alpha)
    moving_feature = ants.iMath(mi, "Laplacian",alpha)

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



def log_feature(fi,mi,alpha):
    fix= ants.iMath(fi, "Laplacian",alpha)
    mix = ants.iMath(mi, "Laplacian",alpha)
    return(fix,mix)
    

# [testlog,testtx] = log_process(fi,mi)
# ants.image_write(testlog, 'testlog.nii.gz')
