"""
Module ``datasets.mnist`` gives access to the binarized version of the
MNIST dataset.  

The original dataset from http://yann.lecun.com/exdb/mnist/ has been
preprocessed so that the inputs are between 0 and 1. The original
training set also has been separated into a new training set and a
validation set.

| **References:**
| The MNIST database of handwritten digits
| LeCun and Cortes
| http://yann.lecun.com/exdb/mnist/
| Classification using Discriminative Restricted Boltzmann Machines (for the train/valid split)
| Larochelle and Bengio
| http://www.cs.toronto.edu/~larocheh/publications/icml-2008-discriminative-rbm.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False):
    """
    Loads the MNIST dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs have been normalized between 0 and 1.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=784
    targets = set(range(10))
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]),int(tokens[-1]))
        #return mlio.libsvm_load_line(line,float,int,sparse,input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'mnist_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [50000,10000,10000]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt',os.path.join(dir_path,'mnist_train.txt'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt',os.path.join(dir_path,'mnist_valid.txt'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt',os.path.join(dir_path,'mnist_test.txt'))
    print 'Done                     '
