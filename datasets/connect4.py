import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False,dtype=np.float64):
    """
    Loads the Connect-4 dataset.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Defined metadata: 
    - 'input_size'
    - 'targets'
    - 'length'

    Reference: Tractable Multivariate Binary Density Estimation and the Restricted Boltzmann Forest
               Larochelle, Bengio and Turian
               link: http://www.cs.toronto.edu/~larocheh/publications/NECO-10-09-1100R2-PDF.pdf
    """
    
    input_size=126
    dir_path = os.path.expanduser(dir_path)
    targets = set([0,1,2])
    target_mapping = {'-1':0,'0':1,'1':2}
    def convert_target(target):
        return target_mapping[target]

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'connect-4_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [16000,4000,47557]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[dtype,int],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/connect-4/connect-4_train.libsvm',os.path.join(dir_path,'connect-4_train.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/connect-4/connect-4_valid.libsvm',os.path.join(dir_path,'connect-4_valid.libsvm'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/connect-4/connect-4_test.libsvm',os.path.join(dir_path,'connect-4_test.libsvm'))
    print 'Done                     '
