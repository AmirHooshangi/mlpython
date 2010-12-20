"""
Module ``datasets.mushrooms`` gives access to the Mushrooms dataset.

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
    Loads the Mushrooms dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of
    data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=112
    dir_path = os.path.expanduser(dir_path)
    targets = set([0,1])
    target_mapping = {'1':0,'2':1}
    def convert_target(target):
        return target_mapping[target]

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'mushrooms_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [2000,500,5624]
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
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mushrooms/mushrooms_train.libsvm',os.path.join(dir_path,'mushrooms_train.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mushrooms/mushrooms_valid.libsvm',os.path.join(dir_path,'mushrooms_valid.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mushrooms/mushrooms_test.libsvm',os.path.join(dir_path,'mushrooms_test.libsvm'))
    print 'Done                     '
