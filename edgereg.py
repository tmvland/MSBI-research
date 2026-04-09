

import ants
import cv2
import numpy as np
from edgemap import make_edgemap

from matplotlib import pyplot as plt

#ref: https://antspy.readthedocs.io/en/latest/registration.html

fi = cv2.imread('fixa.png', cv2.IMREAD_GRAYSCALE)
mi = cv2.imread('mova.png', cv2.IMREAD_GRAYSCALE)


print(fi.shape)

[edgefi,logfi,mri_slicefi] = make_edgemap(fi)
[edgemi,logmi,mri_slicemi] = make_edgemap(mi)

#resizing sCT image to make sure it's interpreted correctly
height, width = fi.shape[:2]
mi= cv2.resize(mi, (width, height), interpolation=cv2.INTER_LINEAR)

fia = ants.from_numpy(fi.astype('uint8'))
fiedge = ants.from_numpy(logfi.astype('uint8'))
mia = ants.from_numpy(mi.astype('uint8'))

#registering the edge images to eachother is resulting in too much warping :(
#trying new method-- register bulk mri to itself and THEN register to edges?
ants.plot(fia, mia, overlay_alpha=0.5)

b = 200


fia = ants.resample_image(fia, (b,b), 1, 0)
mia = ants.resample_image(mia, (b,b), 1, 0)

mytx = ants.registration(fixed=fia, moving=mia, type_of_transform = 'SyNAggro' )

# ants.plot(mi, cmap='Reds', alpha=0.5)
# ants.plot(mytx['warpedmovout'], cmap='Blues')

ants.plot(fia, mia, overlay_alpha=0.5)

ants.plot(fia, mytx['warpedmovout'], overlay_alpha=0.5)

#second registration to edge map

newmove = mytx['warpedmovout']

mytx2 = ants.registration(fixed=fiedge, moving=newmove, type_of_transform = 'SyNAggro' )

ants.plot(fiedge, newmove, overlay_alpha=0.5)

ants.plot(fiedge, mytx2['warpedmovout'], overlay_alpha=0.5)




#extracting deformation matrix
deformation_field = mytx['fwdtransforms'][0]
transform_params = mytx['fwdtransforms']

#applying to other image...
# transformedimg = ants.applytransforms(fixed=fia,moving=mia,transformlist=transformparams)

#getting jacobian as well
jac = ants.create_jacobian_determinant_image(fia,mytx['fwdtransforms'][0],1)
