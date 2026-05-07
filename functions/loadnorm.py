#loadnorm

import ants
import numpy as np
import matplotlib.pyplot as plt
from functions.loadct import loadct, normct, regsct


def loadnorm(fixed_directory, move_directory,lower, upper):

    loadct(move_directory,'output/sctnifti.nii.gz')
    mi = normct('output/sctnifti.nii.gz',lower,upper)
    fi = ants.image_read(fixed_directory)

    [fi, mi, initx] = regsct(fi,mi)


    return(fi,mi,initx)

