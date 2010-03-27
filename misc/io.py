import cPickle,os
import numpy as np
import scipy.io
from gzip import GzipFile as gfile

# This module includes useful functions for loading and saving datasets or objects in general

# Functions to load datasets in different formats.
# Those functions output a pair (data,metadata), where 
# metadata is a dictionary.

### Some helper classes ###

class IteratorWithFields():
    """
    An iterator over the rows of a Numpy array, which separates each row into fields (segments)

    This class helps avoiding the creation of a list of arrays.
    The fields are defined by a list of pairs (beg,end), such that 
    data[:,beg:end] is a field.
    """

    def __init__(self,data,fields):
        self.data = data
        self.fields = fields

    def __iter__(self):
        for r in self.data:
            yield [r[beg:end] for (beg,end) in self.fields ]


class MemoryDataset():
    """
    An iterator over some data, but that puts the content 
    of the data in memory in Numpy arrays.

    Option 'field_shapes' is a list of tuples, corresponding
    to the shape of each fields.

    Optionally, the length of the dataset can also be
    provided. If not, it will be figured out automatically.
    """

    def __init__(self,data,field_shapes,length=None):
        self.data = data
        self.field_shapes = field_shapes
        self.n_fields = len(field_shapes)
        self.mem_data = []
        if length == None:
            # Figure out length
            length = 0
            for example in data:
                length += 1
        self.length = length
        for i in range(self.n_fields):
            sh = field_shapes[i]
            if sh == (1,):
                mem_shape = (length,) # Special case of non-array fields. This will 
                                      # ensure that a non-array field is yielded
            else:
                mem_shape = (length,)+sh
            self.mem_data += [np.zeros(mem_shape)]

        # Put data in memory
        t = 0
        for example in data:
            for i in range(self.n_fields):
                self.mem_data[i][t] = example[i]
            t+=1

    def __iter__(self):
        for t in range(self.length):
            yield [ m[t] for m in self.mem_data ]

class FileDataset():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option 'load_line' is a function which, given 
    a string (a line in the file) outputs an example.
    """

    def __init__(self,filename,load_line):
        self.filename = filename
        self.load_line = load_line

    def __iter__(self):
        stream = open(os.path.expanduser(self.filename))
        for line in stream:
            yield self.load_line(line)
        stream.close()


### ASCII format ###

def ascii_load(filename, convert_input=float, last_column_is_target = False, convert_target=float):
    """
    Reads an ascii file and returns its data and metadata.

    Data can either be a simple numpy array (matrix), or an iterator over (numpy array,target)
    pairs if the last column of the ascii file is to be considered a target.

    Options 'convert_input' and 'convert_target' are functions which must convert
    an element of the ascii file from the string format to the desired format (default: float).

    Defined metadata: 
    - 'input_size'

    """

    f = open(os.path.expanduser(filename))
    lines = f.readlines()

    if last_column_is_target == 0:
        data = np.array([ [ convert_input(i) for i in line.split() ] for line in lines])
        return (data,{'input_size':data.shape[1]})
    else:
        data = np.array([ [ convert_input(i) for i in line.split()[:-1] ] + [convert_target(line.split()[-1])] for line in lines])
        return (IteratorWithFields(data,[(0,data.shape[1]-1),(data.shape[1]-1,data.shape[1])]),
                {'input_size':data.shape[1]-1})
    f.close()

### LIBSVM format ###

def libsvm_load_line(line,convert_non_digit_features=float,convert_target=str,sparse=True,input_size=-1):
    """
    Converts a line (string) of a libsvm file into an example (list).

    This function is used by libsvm_load().
    If sparse is False, option 'input_size' is used to determine the size 
    of the returned 1D array  (it must be big enough to fit all features).
    """
    line = line.strip()
    tokens = line.split()

    # Remove indices < 1
    n_removed = 0
    n_feat = 0
    for token,i in zip(tokens, range(len(tokens))):
        if token.find(':') >= 0:
            if token[:token.find(':')].isdigit():
                if int(token[:token.find(':')]) < 1: # Removing feature ids < 1
                    del tokens[i-n_removed]
                    n_removed += 1
                else:
                    n_feat += 1
        
    if sparse:
        inputs = np.zeros((n_feat))
        indices = np.zeros((n_feat),dtype='int')
    else:
        input = np.zeros((input_size))
    extra = []

    i = 0
    for token in tokens[1:]:
        id_str,input_str = token.split(':')
        if id_str.isdigit():
            if sparse:
                indices[i] = int(id_str)
                inputs[i] = float(input_str)
            else:
                input[int(id_str)-1] = float(input_str)
            i += 1
        else:
            extra += [convert_non_digit_features(id_str,input_str)]
            
    if sparse:
        example = [(inputs, indices), tokens[0]]
    else:
        example = [input,convert_target(tokens[0])]
    if extra:
        example += extra
    return example

def libsvm_load(filename,convert_non_digit_features=float,convert_target=str,sparse=True):
    """
    Reads a LIBSVM file and returns the list of all examples (data) and metadata information.

    In general, each example in the list is a two items list [input, target] where
    - if sparse is True, input is a pair (values, indices) of two vectors 
      (vector of values and of indices). Indices start at 1;
    - if sparse is False, input is a 1D array such that its elements
      at the positions given by indices-1 are set to the associated values, and the
      other elemnents are 0;
    - target is a string corresponding to the target to predict.

    If a 'feature:value' pair is such that 'feature' is not an integer, 
    'value' will be converted to the desired format using option
    'convert_non_digit_features'. This option must be a callable function
    taking 2 string arguments, and will be called as follows:
         output = convert_non_digit_features(feature_str,value_str)
    where 'feature_str' and 'value_str' are 'feature' and 'value' in string format.
    Its output will be appended to the list of the given example.

    Defined metadata: 
    - 'targets'
    - 'input_size'

    """

    stream = open(os.path.expanduser(filename))
    data = []
    metadata = {}
    targets = set()
    input_size = 0
    for line in stream:
        example = libsvm_load_line(line,convert_non_digit_features,convert_target,True)
        max_non_zero_feature = max(example[0][1])
        if max_non_zero_feature > input_size:
            input_size = max_non_zero_feature
        targets.add(example[1])
        # If not sparse, first pass through libsvm file just 
        # figures out the input_size and targets
        if sparse:
            data += [example]
    stream.close()

    if not sparse:
        # Now that we know the input_size, we can load the data
        stream = open(os.path.expanduser(filename))
        for line in stream:
            example = libsvm_load_line(line,convert_non_digit_features,convert_target,False,input_size)
            data += [example]
        stream.close()
        
    metadata['targets'] = targets
    metadata['input_size'] = input_size
    return data, metadata

### Generic save/load functions, using cPickle ###

def save(p, filename):
    f=file(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def load(filename): 
    f=file(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y

def gsave(p, filename):
    f=gfile(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def gload(filename):
    f=gfile(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y


### For loading large datasets which don't fit in memory ###

def load_line_default(line):
    return np.array([float(i) for i in line.split()]) # Converts each element to a float

def load_from_file(filename,load_line=load_line_default):
    """
    Loads a dataset from a file, without loading it in memory.
    """
    return FileDataset(filename,load_line)
    
