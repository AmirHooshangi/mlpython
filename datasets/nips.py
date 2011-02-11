"""
Module ``datasets.nips`` gives access to the NIPS 0-12 dataset.

| **Reference:** 
| Tractable Multivariate Binary Density Estimation and the Restricted Boltzmann Forest
| Larochelle, Bengio and Turian
| http://www.cs.toronto.edu/~larocheh/publications/NECO-10-09-1100R2-PDF.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the NIPS 0-12 dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'length'``

    """
    
    input_size=500
    dir_path = os.path.expanduser(dir_path)
    def load_line(line):
        tokens = line.split()
        return np.array([int(i) for i in tokens[:-1]]) #The last element is bogus (don't ask...)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'nips-0-12_all_shuffled_bidon_target_' + ds + '.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [400,100,1240]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,)],[np.float64],l) for d,l in zip([train,valid,test],lengths)]
        
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
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/nips-0-12/nips-0-12_all_shuffled_bidon_target_train.amat',os.path.join(dir_path,'nips-0-12_all_shuffled_bidon_target_train.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/nips-0-12/nips-0-12_all_shuffled_bidon_target_valid.amat',os.path.join(dir_path,'nips-0-12_all_shuffled_bidon_target_valid.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/nips-0-12/nips-0-12_all_shuffled_bidon_target_test.amat',os.path.join(dir_path,'nips-0-12_all_shuffled_bidon_target_test.amat'))
    print 'Done                     '
