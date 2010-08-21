import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,sparse=False,load_to_memory=False,dtype=np.float64):
    """
    Loads the OCR letter dataset.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Defined metadata: 
    - 'input_size'
    - 'length'

    """
    
    input_size=128
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        return mlio.libsvm_load_line(line,float,int,sparse,input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'ocr_letter_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    if load_to_memory:
        lengths = [32152,10000,10000]
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],dtype,l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    dir_path = os.path.expanduser(dir_path)
    train_file,valid_file,test_file = [os.path.join(dir_path, 'ocr_letter_' + ds + '.txt') for ds in ['train','valid','test']]
    try:

        print 'TODO!'
        #file = open(train_file)
        #n_queries = 0
        #small_train_file = os.path.join(dir_path, 'set1.small_train.txt')
        #small_valid_file = os.path.join(dir_path, 'set1.small_valid.txt')
        #train_file = open(small_train_file,'w')
        #valid_file = open(small_valid_file,'w')
        #qid_split = 13944
        #print 'Seperating training into smaller training/validation sets'
        #qid = 0
        #for line in file:
        #    new_qid = int(line.split('qid:')[1].split(' ')[0])
        #    if qid != new_qid:
        #        print '...reading query %i\r' % new_qid,
        #    qid = new_qid
        #    if qid <= qid_split:
        #        train_file.write(line)
        #    else:
        #        valid_file.write(line)
        #train_file.close()
        #valid_file.close()

        print 'Done                     '
    except IOError:
        print 'Could not retrieve and format the data somehow.'
