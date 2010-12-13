"""
Module ``datasets.newsgroups`` gives access to the 20-newsgroups dataset.

The original dataset from
http://people.csail.mit.edu/jrennie/20Newsgroups/ has been
preprocessed to limit the vocabulary to the 5000 most frequent
words. The binary bag-of-word representation has then been computed
for each document. The original training set also has been separated
into a new training set and a validation set.


| **Reference:**
| 20 Newsgroups (web page where the original dataset was obtained)
| Jason Rennie
| http://people.csail.mit.edu/jrennie/20Newsgroups/
| Classification using Discriminative Restricted Boltzmann Machines (for the train/valid split and preprocessing)
| Larochelle and Bengio
| http://www.cs.toronto.edu/~larocheh/publications/icml-2008-discriminative-rbm.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False,dtype=np.float64):
    """
    Loads the 20-newsgroups dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs have been put in binary format, and the vocabulary has been
    restricted to 5000 words.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=5000
    targets = set(range(20))
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]),int(tokens[-1]))

    train_file,valid_file,test_file = [os.path.join(dir_path, '20newsgroups_' + ds + '_binary_5000_voc.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [9578,1691,7505]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[dtype,int],l) for d,l in zip([train,valid,test],lengths)]
        
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
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/20newsgroups/20newsgroups_train_binary_5000_voc.txt',os.path.join(dir_path,'20newsgroups_train_binary_5000_voc.txt'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/20newsgroups/20newsgroups_valid_binary_5000_voc.txt',os.path.join(dir_path,'20newsgroups_valid_binary_5000_voc.txt'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/20newsgroups/20newsgroups_test_binary_5000_voc.txt',os.path.join(dir_path,'20newsgroups_test_binary_5000_voc.txt'))
    print 'Done                     '
