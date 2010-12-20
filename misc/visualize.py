"""
Module ``misc.visualize`` includes useful functions 
for visualizing datasets or image filters.

This module contains the following functions:

* ``show_filters``:        plots the filters in a weight matrix.
* ``show_binary_images``:  plots samples from a dataset of images with binary pixels.
* ``show_color_images``:   plots samples from a dataset of images with color pixels.

"""

import numpy, sys, random
from matplotlib import cm
from matplotlib.pylab import figure, imshow, show, xticks, yticks, array

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
    if samples.shape[0] < nrows*ncols:
        samples_padded = numpy.zeros((nrows*ncols,samples.shape[1]))
        samples_padded[:samples.shape[0],:] = samples
        samples = samples_padded

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

def show_color_images(samples, nsamples, d1, d2, nrows, ncols):
    """
    Plots samples in a Numpy 2D array ``samples`` as ``d1`` by ``d2`` images.
    (one sample per row of ``samples``).

    The samples are assumed to be color images. The first ``d1*d2``
    elements of each row are the R channel values of each pixel, then
    follows the G and B channels. The images are layed out in a
    ``nrows`` by ``ncols`` grid.

    Thanks to Ilya Sutskever for sharing his code, from which this
    code is inspired.
    """
    samples = samples[:nsamples,:]

    def fix(X):
        Y = X - X.min()
        Y /= Y.max()
        return Y
    
    def print_aligned(w):
        n1 = nrows
        n2 = ncols
        r1 = d1
        r2 = d2
        Z = numpy.zeros(((r1+1)*n1, (r1+1)*n2), 'd')
        i1, i2 = 0, 0
        for i1 in range(n1):
            for i2 in range(n2):
                i = i1*n2+i2
                if i>=w.shape[1]: break
                Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
        return Z

    R = samples[:,:(d1*d2)]
    G = samples[:,(d1*d2):(2*d1*d2)]
    B = samples[:,(2*d1*d2):(3*d1*d2)]
    
    img = array([print_aligned(R.T), 
                 print_aligned(G.T), 
                 print_aligned(B.T)]).transpose([1, 2, 0])

    imshow(fix(img), interpolation='nearest')
    xticks([])
    yticks([])
    show()
