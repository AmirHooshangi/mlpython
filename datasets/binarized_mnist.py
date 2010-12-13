"""
Module ``datasets.binarized_mnist`` gives access to the binarized version of the MNIST dataset.

| **References:**
| On the Quantitative Analysis of Deep Belief Networks
| Salakhutdinov and Murray
| http://www.mit.edu/~rsalakhu/papers/dbn_ais.pdf
| The MNIST database of handwritten digits
| LeCun and Cortes
| http://yann.lecun.com/exdb/mnist/

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False,dtype=np.float64):
    """
    Loads a binarized version of MNIST. 

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'input_size'``
    * ``'length'``

    """
    
    input_size=784
    dir_path = os.path.expanduser(dir_path)
    def load_line(line):
        tokens = line.split()
        return np.array([int(i) for i in tokens])

    train_file,valid_file,test_file = [os.path.join(dir_path, 'binarized_mnist_' + ds + '.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [50000,10000,10000]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,)],[dtype],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',os.path.join(dir_path,'binarized_mnist_train.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',os.path.join(dir_path,'binarized_mnist_valid.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',os.path.join(dir_path,'binarized_mnist_test.amat'))

    print 'Done                     '
