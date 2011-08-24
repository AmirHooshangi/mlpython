"""
Module ``datasets.scene`` gives access to the Scene dataset.

| **Reference:** 
| Learning multi-label scene classification
| Boutell, Luo, Shen, Brown
| http://www.sciencedirect.com/science?_ob=MImg&_imagekey=B6V14-4CF14JX-1-5D&_cdi=5664&_user=994540&_pii=S0031320304001074&_origin=search&_coverDate=09%2F30%2F2004&_sk=999629990&view=c&wchp=dGLzVzb-zSkzk&md5=de064f2a6ecb558f52afb78992218296&ie=/sdarticle.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Scene dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size=294
    target_size=6
    dir_path = os.path.expanduser(dir_path)
    def convert_target(target_str):
        targets = np.zeros((target_size))
        for l in target_str.split(','):
            id = int(l)
            targets[id] = 1
        return targets

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'scene_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [1000,211,1196]
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
    urllib.urlretrieve('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/scene_train.bz2',os.path.join(dir_path,'scene_train.bz2'))
    urllib.urlretrieve('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/scene_test.bz2',os.path.join(dir_path,'scene_test.bz2'))

    import bz2
    train_valid_bz2_file = bz2.BZ2File(os.path.join(dir_path,'scene_train.bz2'))
    test_bz2_file = bz2.BZ2File(os.path.join(dir_path,'scene_test.bz2'))

    print 'Splitting training set into smaller training/validation sets'
    train_file,valid_file,test_file = [open(os.path.join(dir_path, 'scene_' + ds + '.libsvm'),'w') for ds in ['train','valid','test']]

    # Putting train/valid data in memory
    train_valid_data = [ line for line in train_valid_bz2_file ]

    # Shuffle data
    import random
    random.seed(12345)
    perm = range(len(train_valid_data))
    random.shuffle(perm)
    line_id = 0
    train_valid_split = 1000
    for i in perm:
        s = train_valid_data[i]
        if line_id < train_valid_split:
            train_file.write(s)
        else:
            valid_file.write(s)
        line_id += 1
    train_file.close()
    valid_file.close()
    train_valid_bz2_file.close()

    for line in test_bz2_file:
        test_file.write(line)

    test_file.close()
    test_bz2_file.close()
    print 'Done                     '
