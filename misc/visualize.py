"""
Module ``misc.visualize`` includes useful functions 
for visualizing datasets or image filters.

This module contains the following functions:

* ``show_filters``:        plots the filters in a weight matrix.
* ``show_binary_images``:  plots samples from a dataset of images with binary pixels.

"""

import numpy, sys, random
from matplotlib import cm
from matplotlib.pylab import figure, imshow, show, xticks, yticks

def show_filters(weights,nweights,d1, d2, nrows, ncols, scale):
    """
    Plots the rows of Numpy 2D array ``weights`` as ``d1`` by ``d2`` images.

    The images are layed out in a ``nrows`` by ``ncols`` grid.

    Option ``scale`` sets the maximum absolute value of elements in ``weights``
    that will be plotted (larger values will be clamped to ``scale``, with the
    right sign).
    """
    perm = range(nweights)
    #random.shuffle(perm)
    image = -scale*numpy.ones((nrows*(d1+1)-1,ncols*(d2+1)-1),dtype=float)
    for i in range(nrows):
        for j in range(ncols):
            image[(i*d1+i):((i+1)*d1+i),(j*d2+j):((j+1)*d2+j)] = -1*weights[perm[i*ncols + j]].reshape(d1,d2)

    for i in range(nrows*(d1+1)-1):
        for j in range(ncols*(d2+1)-1):
            a = image[i,j]
            if a > scale:
                image[i,j] = scale
            if a < -scale:
                image[i,j] = -scale

    bordered_image = scale * numpy.ones((nrows*(d1+1)+1,ncols*(d2+1)+1),dtype=float)

    bordered_image[1:nrows*(d1+1),1:ncols*(d2+1)] = image

    imshow(bordered_image,cmap = cm.Greys,interpolation='nearest')
    xticks([])
    yticks([])
    show()

def show_binary_images(samples, nsamples, d1, d2, nrows, ncols):
    """
    Plots samples in a Numpy 2D array ``samples`` as ``d1`` by ``d2`` images.
    (one sample per row of ``samples``).

    The samples are assumed to be images with binary pixels. The
    images are layed out in a ``nrows`` by ``ncols`` grid.
    """
    perm = range(nsamples)
    #random.shuffle(perm)
    image = 0.5*numpy.ones((nrows*(d1+1)-1,ncols*(d2+1)-1),dtype=float)
    for i in range(nrows):
        for j in range(ncols):
            image[(i*d1+i):((i+1)*d1+i),(j*d2+j):((j+1)*d2+j)] = (1-samples[perm[i*ncols + j]].reshape(d1,d2))

    bordered_image = 0.5 * numpy.ones((nrows*(d1+1)+1,ncols*(d2+1)+1),dtype=float)

    bordered_image[1:nrows*(d1+1),1:ncols*(d2+1)] = image

    imshow(bordered_image,cmap = cm.Greys,interpolation='nearest')
    xticks([])
    yticks([])
    show()
