import mlpython.misc.io as mlio

def load(dir_path,set_id=1,sparse=False):
    """
    Loads the Yahoo! Learning to Rank Challenge data.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Option 'set_id' determines the set that is loaded (1 or 2, default is 1).

    Note that, because the data is quite big, it is not loaded in memory and is
    instead always read directly from the associated files.

    Defined metadata: 
    - 'input_size'
    - 'scores'
    - 'n_queries'
    - 'length'

    """
    
    input_size=700

    def convert(feature,value):
        if feature is not 'qid':
            raise ValueError('Unexpected feature')
        return int(value)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert,sparse,input_size)

    # Get data file paths
    train_file,valid_file,test_file = [dir_path + '/' + 'set' + str(set_id) + '.' + ds + '.txt' for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    # Get metadata
    if set_id == 1:
        n_queries = [19944,2994,6983]
        lengths = [473134,71083,165660]
    else:
        n_queries = [1266,1266,3798]
        lengths = [34815,34881,103174]

    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                        'scores':range(5),
                                        'n_queries':nq,
                                        'length':l} for nq,l in [n_queries,lengths]]

    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    print 'Go to http://learningtorankchallenge.yahoo.com/ to download the data.'
