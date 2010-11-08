import numpy, sys, random
from matplotlib import cm
from matplotlib.pylab import figure, imshow, show, xticks, yticks

def show_filters(weights,nweights,d1, d2, nrows, ncols, scale):
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

def show_binary_images(samples,nsamples,d1, d2, nrows, ncols):
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
