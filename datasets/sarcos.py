"""
Module ``datasets.sarcos`` gives access to the SARCOS dataset.

This is a multi-dimensional regression dataset, with outputs in [0,1].
The task is an inverse dynamics problem for a seven degrees-of-freedom
SARCOS anthropomorphic robot arm.

The inputs have varying range, so PCA is recommended.

| **References:**
| LWPR: An O(n) Algorithm for Incremental Real Time Learning in High Dimensional Space
| Vijayakumar and Schaal
| http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.4252&rep=rep1&type=pdf
|
| The Gaussian Processes Web Site
| http://www.gaussianprocess.org/gpml/data/

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    SARCOS inverse dynamics dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size=21
    target_size=7
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:input_size]]), np.array([float(i) for i in tokens[input_size:]]))

    train_file,valid_file,test_file = [os.path.join(dir_path, 'sarcos_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [40036,4448,4449]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(target_size,)],[np.float64,np.float64],l) for d,l in zip([train,valid,test],lengths)]
        
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
    urllib.urlretrieve('http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat',os.path.join(dir_path,'sarcos_inv.mat'))
    urllib.urlretrieve('http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat',os.path.join(dir_path,'sarcos_inv_test.mat'))

    # Writing everything into text files, to allow for not loading the data into memory
    def write_to_txt_file(mat,filename):
        f = open(filename,'w')
        for mat_i in mat:
            line = ' '.join(['%.6f' % mat_ij for mat_ij in mat_i]) + '\n'
            f.write(line)
        f.close()

    import scipy.io
    train_valid_set = scipy.io.loadmat(os.path.join(dir_path,'sarcos_inv.mat'))['sarcos_inv']
    valid_size = round(0.1*len(train_valid_set))
    train_size = len(train_valid_set) - valid_size

    import random
    random.seed(12345)
    perm = range(len(train_valid_set))
    random.shuffle(perm)
    train_valid_set = train_valid_set[perm,:]

    train_set = train_valid_set[:train_size,:]
    valid_set = train_valid_set[train_size:,:]
    test_set = scipy.io.loadmat(os.path.join(dir_path,'sarcos_inv_test.mat'))['sarcos_inv_test']

    write_to_txt_file(train_set,os.path.join(dir_path,'sarcos_train.txt'))
    write_to_txt_file(valid_set,os.path.join(dir_path,'sarcos_valid.txt'))
    write_to_txt_file(test_set,os.path.join(dir_path,'sarcos_test.txt'))

    print 'Done                     '
