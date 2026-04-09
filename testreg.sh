

import ants
import cv2
import numpy as np

from matplotlib import pyplot as plt

fi = cv2.imread('fixa.png', cv2.IMREAD_GRAYSCALE)
mi = cv2.imread('mova.png', cv2.IMREAD_GRAYSCALE)


print(fi.shape)

fia = ants.from_numpy(fi.astype('uint8'))
mia = ants.from_numpy(mi.astype('uint8'))


ants.plot(fia, mia, overlay_alpha=0.5)


fia = ants.resample_image(fia, (200,200), 1, 0)
mia = ants.resample_image(mia, (200,200), 1, 0)

mytx = ants.registration(fixed=fia, moving=mia, type_of_transform = 'SyN' )
mytx = ants.registration(fixed=fia, moving=mia, type_of_transform = 'antsRegistrationSyN[t]' )
mytx = ants.registration(fixed=fia, moving=mia, type_of_transform = 'antsRegistrationSyN[b]' )
mytx = ants.registration(fixed=fia, moving=mia, type_of_transform = 'antsRegistrationSyN[s]' )



# ants.plot(mi, cmap='Reds', alpha=0.5)
# ants.plot(mytx['warpedmovout'], cmap='Blues')

ants.plot(fia, mia, overlay_alpha=0.5)

ants.plot(fia, mytx['warpedmovout'], overlay_alpha=0.5)

#extracting deformation matrix
deformation_field = mytx['fwdtransforms'][0]
transform_params = mytx['fwdtransforms']

#applying to other image...
# transformedimg = ants.applytransforms(fixed=fia,moving=mia,transformlist=transformparams)

#getting jacobian as well
jac = ants.create_jacobian_determinant_image(fia,mytx['fwdtransforms'][0],1)
