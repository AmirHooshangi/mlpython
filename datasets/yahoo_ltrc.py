import mlpython.misc.io as mlio
import os

def load(dir_path,set_id=0,sparse=False,load_to_memory=False):
    """
    Loads the Yahoo! Learning to Rank Challenge data.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Option 'set_id' determines the set that is loaded (0, 1 or 2, default is 0).

    Set 0 is a "home made" train/valid split of the original training set, 
    required since only the training set is labeled (until all the data is
    released). Not test set is generated for that purpose.

    Note that, because the data is quite big, it is not loaded in memory and is
    instead always read directly from the associated files.

    Defined metadata: 
    - 'input_size'
    - 'scores'
    - 'n_queries'
    - 'length'

    """
    
    input_size=700
    dir_path = os.path.expanduser(dir_path)

    def convert(feature,value):
        if feature != 'qid':
            raise ValueError('Unexpected feature')
        return int(value)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert,int,sparse,input_size)

    if set_id == 0:
        n_queries = [13944,6000]
        lengths = [294336,178798]

        train_file,valid_file = [os.path.join(dir_path, 'set1.' + ds + '.txt') for ds in ['small_train','small_valid']]
        # Get data
        train,valid = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file]]

        if load_to_memory:
            train,valid = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],l) for d,l in zip([train,valid],lengths)]

        # Get metadata
        train_meta,valid_meta = [{'input_size':input_size,
                                  'scores':range(5),
                                  'n_queries':nq,
                                  'length':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta)}
    else:
        if set_id == 1:
            n_queries = [19944,2994,6983]
            lengths = [473134,71083,165660]
        else:
            n_queries = [1266,1266,3798]
            lengths = [34815,34881,103174]

        # Get data file paths
        train_file,valid_file,test_file = [os.path.join(dir_path, 'set' + str(set_id) + '.' + ds + '.txt') for ds in ['train','valid','test']]
        # Get data
        train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
        if load_to_memory:
            train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],l) for d,l in zip([train,valid,test],lengths)]

        train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                            'scores':range(5),
                                            'n_queries':nq,
                                            'length':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    dir_path = os.path.expanduser(dir_path)
    train_file = os.path.join(dir_path, 'set1.train.txt')
    try:
        file = open(train_file)
        n_queries = 0
        small_train_file = os.path.join(dir_path, 'set1.small_train.txt')
        small_valid_file = os.path.join(dir_path, 'set1.small_valid.txt')
        train_file = open(small_train_file,'w')
        valid_file = open(small_valid_file,'w')
        qid_split = 13944
        print 'Seperating training into smaller training/validation sets'
        qid = 0
        for line in file:
            new_qid = int(line.split('qid:')[1].split(' ')[0])
            if qid != new_qid:
                print '...reading query %i\r' % new_qid,
            qid = new_qid
            if qid <= qid_split:
                train_file.write(line)
            else:
                valid_file.write(line)

        print 'Done                     '
    except IOError:
        print 'Go to http://learningtorankchallenge.yahoo.com/ to download the data.'
        print 'Once this is done, call this function again.'
