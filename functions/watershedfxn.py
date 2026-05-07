import ants
import numpy as np
import matplotlib.pyplot as plt
from functions.loadct import loadct, normct, regsct
import os
import pydicom
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def watershed_process(fi,mi):
    fi_denoised = ants.denoise_image(fi)
    mi_denoised = ants.denoise_image(mi)

    print('img denoised')
    # convert to numpy for watershed
    img_arr_fi = fi_denoised.numpy()
    img_arr_mi = mi_denoised.numpy()

    thresholdfi = np.mean(img_arr_fi)
    thresholdmi = np.mean(img_arr_mi)

    local_maximafi = img_arr_fi.max()
    local_maximami = img_arr_mi.max()
    print(local_maximafi,local_maximami)
    print('extrema identified')

    #create markers (e.g., using local minima/maxima or distance transform)

    distancefi = ndi.distance_transform_edt(img_arr_fi > thresholdfi)
    distancemi = ndi.distance_transform_edt(img_arr_mi > thresholdmi)

    print('markers created')

    markersfi = ndi.label(local_maximafi)[0]
    markersmi = ndi.label(local_maximami)[0]


    #running watershed
    labelsfi = watershed(-distancefi, markersfi, mask=(img_arr_fi>thresholdfi))
    labelsmi = watershed(-distancemi, markersmi, mask=(img_arr_mi>thresholdmi))

    #convert back to ANTsImage
    fixed_feature = ants.from_numpy(labelsfi.astype('float32'), origin=fi.origin, 
                                 spacing=fi.spacing, direction=fi.direction)

    moving_feature = ants.from_numpy(labelsmi.astype('float32'), origin=mi.origin, 
                                 spacing=mi.spacing, direction=mi.direction)

    print('watershed run')

    #deformable registration based on watershed segmentation
    mytx = ants.registration(
        fixed=fi, 
        moving=mi, 
        type_of_transform='SyN',
        multivariate_extras=[('MeanSquares', fixed_feature, moving_feature, 0.5, 0.0)]
    )


    registered_img = ants.apply_transforms(fixed=fi, moving=mi,
                                      transformlist=mytx['fwdtransforms'])
    
    return(registered_img,mytx)


def watershed_feature(fi,mi):
    fi_denoised = ants.denoise_image(fi)
    mi_denoised = ants.denoise_image(mi)

    # convert to numpy for watershed
    img_arr_fi = fi_denoised.numpy()
    img_arr_mi = mi_denoised.numpy()

    thresholdfi = np.mean(img_arr_fi)
    thresholdmi = np.mean(img_arr_mi)

    local_maximafi = img_arr_fi.max()
    local_maximami = img_arr_mi.max()

    #create markers (e.g., using local minima/maxima or distance transform)

    distancefi = ndi.distance_transform_edt(img_arr_fi > thresholdfi)
    distancemi = ndi.distance_transform_edt(img_arr_mi > thresholdmi)


    markersfi = ndi.label(local_maximafi)[0]
    markersmi = ndi.label(local_maximami)[0]


    #running watershed
    labelsfi = watershed(-distancefi, markersfi, mask=(img_arr_fi>thresholdfi))
    labelsmi = watershed(-distancemi, markersmi, mask=(img_arr_mi>thresholdmi))

    #convert back to ANTsImage
    fix= ants.from_numpy(labelsfi.astype('float32'), origin=fi.origin, 
                                 spacing=fi.spacing, direction=fi.direction)

    mix = ants.from_numpy(labelsmi.astype('float32'), origin=mi.origin, 
                                 spacing=mi.spacing, direction=mi.direction)
    return(fix,mix)

# [testwtr,testtx] = watershed_process(fi,mi)
# ants.image_write(testwtr, 'testwtr.nii.gz')


