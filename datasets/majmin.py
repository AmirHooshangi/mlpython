"""
Module ``datasets.majmin`` gives access to the MajMin dataset.

| **Reference:** 
| A Web-Based Game for Collecting Music Metadata
| Mandel and Ellis
| http://mr-pc.org/work/jnmr08.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the MajMin dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size=389
    target_size=96
    dir_path = os.path.expanduser(dir_path)
    
    def convert_target(target_str):
        targets = np.zeros((target_size))
        if target_str != '':
            for l in target_str.split(','):
                id = int(l)
                targets[id] = 1
        return targets

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'majmin_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [1587,471,480]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(target_size,)],[np.float64,bool],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'target_size':target_size,
                                        'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/majmin/majmin_train.libsvm',os.path.join(dir_path,'majmin_train.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/majmin/majmin_valid.libsvm',os.path.join(dir_path,'majmin_valid.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/majmin/majmin_test.libsvm',os.path.join(dir_path,'majmin_test.libsvm'))
    print 'Done                     '
