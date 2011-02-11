"""
Module ``datasets.corrupted_ocr_letters`` gives access to the corrupted
version of the OCR letters dataset.

This is a multilabel classification dataset, with binary targets. The
task is to remove noise from images of 4 characters obtained from the
OCR letters dataset (see ``datasets.ocr_letters``). The noise include
lines crossing the image and single pixels randomly switched to 1.

"""

import mlpython.misc.io as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False):
    """
    Corrupted OCR letters dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs and targets are binary.

    **Defined metadata:**

    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size=16*(32+3)
    target_size=16*(32+3)
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:input_size]]), np.array([float(i) for i in tokens[input_size:]]))

    train_file,valid_file,test_file = [os.path.join(dir_path, 'corrupted_ocr_letters_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [10000,2000,2000]
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
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/corrupted_ocr_train.mat',os.path.join(dir_path,'corrupted_ocr_train.mat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/ocr_train.mat',os.path.join(dir_path,'ocr_train.mat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/corrupted_ocr_valid.mat',os.path.join(dir_path,'corrupted_ocr_valid.mat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/ocr_valid.mat',os.path.join(dir_path,'ocr_valid.mat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/corrupted_ocr_test.mat',os.path.join(dir_path,'corrupted_ocr_test.mat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/corrupted_ocr_letters/ocr_test.mat',os.path.join(dir_path,'ocr_test.mat'))

    # Writing everything into text files, to allow for not loading the data into memory
    def write_to_txt_file(u,v,filename):
        f = open(filename,'w')
        for u_t,v_t in zip(u,v):
            for i in range(len(u_t)):
                f.write(str(int(u_t[i]))+' ')
            for i in range(len(v_t)-1):
                f.write(str(int(v_t[i]))+' ')
            f.write(str(int(v_t[-1]))+'\n')
        f.close()

    import scipy.io
    u = scipy.io.loadmat(os.path.join(dir_path,'corrupted_ocr_train.mat'))['mat']
    v = scipy.io.loadmat(os.path.join(dir_path,'ocr_train.mat'))['mat']
    write_to_txt_file(u,v,os.path.join(dir_path,'corrupted_ocr_letters_train.txt'))

    u = scipy.io.loadmat(os.path.join(dir_path,'corrupted_ocr_valid.mat'))['mat']
    v = scipy.io.loadmat(os.path.join(dir_path,'ocr_valid.mat'))['mat']
    write_to_txt_file(u,v,os.path.join(dir_path,'corrupted_ocr_letters_valid.txt'))

    u = scipy.io.loadmat(os.path.join(dir_path,'corrupted_ocr_test.mat'))['mat']
    v = scipy.io.loadmat(os.path.join(dir_path,'ocr_test.mat'))['mat']
    write_to_txt_file(u,v,os.path.join(dir_path,'corrupted_ocr_letters_test.txt'))

    print 'Done                     '
