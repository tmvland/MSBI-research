import ants
import imageio.v3 as iio

def convertpng(imgname):
    data = iio.imread(imgname)
    if len(data.shape) == 3:
        data = data[:,:,0] 
    antsdata = ants.from_numpy(data.astype('float32'))
    ants.image_write(antsdata, f'{str(imgname)}.nii.gz')

    return()

