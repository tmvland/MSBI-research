import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import skimage as ski
from matplotlib import pyplot as plt

#pulled a lot of code from here:
#https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_morphsnakes.html
#only using MorphACWE-- MorphGAC can't get at any internal details
#Activecontour region growing rather than edge segmentation

def make_regions(fi):

    # Morphological ACWE
    image = fi
    image = image.get_fdata()

    # Initial level set
    init_ls = ski.segmentation.checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = ski.segmentation.morphological_chan_vese(
    image, num_iter=35, init_level_set=init_ls, smoothing=3, iter_callback=callback)

    return ls

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store
