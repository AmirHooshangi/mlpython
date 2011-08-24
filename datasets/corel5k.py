"""
Module ``datasets.corel5k`` gives access to the Corel5k dataset.

| **Reference:** 
| Object recognition as machine translation: Learning a lexicon for a fixed image vocabulary
| Duygulu, Barnard, de Freitas, Forsyth
| http://www.ee.columbia.edu/~sfchang/course/spr/papers/Duygulu-ECCV-02-1.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Corel5k dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size = 499
    target_size = 374
    dir_path = os.path.expanduser(dir_path)

    def convert_target(target_str):
        targets = np.zeros((target_size))
        for l in target_str.split(','):
            id = int(l)
            targets[id] = 1
        return targets

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'corel5k_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [3600,900,500]
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
    urllib.urlretrieve('http://mulan.sourceforge.net/files/Corel5k.rar',os.path.join(dir_path,'corel5k.rar'))

    print 'Splitting dataset into training/validation/test sets'
    start_class_id = 499
    cmd ='unrar e ' + os.path.join(dir_path,'corel5k.rar') + ' ' + dir_path
    os.system(cmd)
    
    def arff_to_libsvm(lines):
        libsvm_lines = []
        i = 0
        while lines[i].strip() != '@data':
            i+=1
        i+=1
        for line in lines[i:]:
            line = line.strip()
            tokens = line.split(',')
            inputs = []
            id = 1 # input id's start at 1
            for tok in tokens[:start_class_id]:
                if tok != '0':
                    inputs += [ str(id)+':'+tok ]
                id += 1
            
            targets = []
            id = 0 # label id's start at 0 (I know, weird...)
            for tok in tokens[start_class_id:]:
                if tok != '0':
                    targets +=  [ str(id) ]
                id += 1
            libsvm_lines += [','.join(targets) + ' ' + ' '.join(inputs) + '\n']
        return libsvm_lines

    f = open(os.path.join(dir_path,'Corel5k-train.arff'))
    train_valid_lines = arff_to_libsvm(f.readlines())
    f.close()

    f = open(os.path.join(dir_path,'Corel5k-test.arff'))
    test_lines = arff_to_libsvm(f.readlines())
    f.close()

    import random
    random.seed(12345)
    random.shuffle(train_valid_lines)
    random.shuffle(test_lines)

    valid_size = int(round(0.2*len(train_valid_lines)))
    train_size = len(train_valid_lines)-valid_size
    train_lines = train_valid_lines[:train_size]
    valid_lines = train_valid_lines[train_size:]

    f = open(os.path.join(dir_path,'corel5k_train.libsvm'),'w')
    for line in train_lines:
        f.write(line)
    f.close()

    f = open(os.path.join(dir_path,'corel5k_valid.libsvm'),'w')
    for line in valid_lines:
        f.write(line)
    f.close()

    f = open(os.path.join(dir_path,'corel5k_test.libsvm'),'w')
    for line in test_lines:
        f.write(line)
    f.close()
    print 'Done                     '
