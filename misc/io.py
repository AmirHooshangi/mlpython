"""
Module ``misc.io`` includes useful functions 
for loading and saving datasets or objects in general

This module contains the following functions:

* ``load_from_file``:        Loads a dataset from a file without allocating memory for it.
* ``ascii_load``:            Reads an ASCII file and returns its data and metadata.
* ``libsvm_load``:           Reads a LIBSVM file and returns its data and metadata.
* ``libsvm_load_line``:      Converts a line from a LIBSVM file in an example.
* ``save``:                  Saves an object into a file.
* ``load``:                  Loads an object from a file.
* ``gsave``:                 Saves an object into a gzipped file
* ``gload``:                 Loads an object from a gzipped file

and the following classes:

* ``IteratorWithFields``:   Iterator which separates the rows of a Numpy array into fields.
* ``MemoryDataset``:        Iterator over some data put in memory as a Numpy array.
* ``FileDataset``:          Iterator over a file whose lines are converted in examples.    

"""

import cPickle,os
import numpy as np
import scipy.io
from gzip import GzipFile as gfile


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
            yield [ (r[beg] if beg+1==end else r[beg:end]) for (beg,end) in self.fields ]


class MemoryDataset():
    """
    An iterator over some data, but that puts the content 
    of the data in memory in Numpy arrays.

    Option ``'field_shapes'`` is a list of tuples, corresponding
    to the shape of each fields.

    Option ``dtypes`` determines the type of each field (float, int, etc.).

    Optionally, the length of the dataset can also be
    provided. If not, it will be figured out automatically.
    """

    def __init__(self,data,field_shapes,dtypes,length=None):
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
            self.mem_data += [np.zeros(mem_shape,dtype=dtypes[i])]

        # Put data in memory
        t = 0
        if self.n_fields == 1:
            for example in data:
                self.mem_data[0][t] = example
                t+=1
        else:
            for example in data:
                for i in range(self.n_fields):
                    self.mem_data[i][t] = example[i]
                t+=1

    def __iter__(self):
        if self.n_fields == 1:
            for example in self.mem_data[0]:
                yield example
        else:
            for t in range(self.length):
                yield [ m[t] for m in self.mem_data ]


class FileDataset():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option ``'load_line'`` is a function which, given 
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

### For loading large datasets which don't fit in memory ###

def load_line_default(line):
    return np.array([float(i) for i in line.split()]) # Converts each element to a float

def load_from_file(filename,load_line=load_line_default):
    """
    Loads a dataset from a file, without loading it in memory.

    It returns an iterator over the examples from that fine. This is based
    on class ``FileDataset``.
    """
    return FileDataset(filename,load_line)
    

# Functions to load datasets in different common formats.
# Those functions output a pair (data,metadata), where 
# metadata is a dictionary.

### ASCII format ###

def ascii_load(filename, convert_input=float, last_column_is_target = False, convert_target=float):
    """
    Reads an ASCII file and returns its data and metadata.

    Data can either be a simple Numpy array (matrix), or an iterator
    over (numpy array,target) pairs if the last column of the ASCII
    file is to be considered a target.

    Options ``'convert_input'`` and ``'convert_target'`` are functions
    which must convert an element of the ASCII file from the string
    format to the desired format.

    **Defined metadata:**

    * ``'input_size'``

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
    Converts a line (string) of a LIBSVM file into an example (list).

    This function is used by ``libsvm_load()``.
    If ``sparse`` is False, option ``'input_size'`` is used to determine the size 
    of the returned 1D array  (it must be big enough to fit all features).
    """
    line = line.rstrip() # Must not remove whitespace on the left, for multi-label datasets
                         # where an empty labels means all labels are 0.
    tokens = line.split(' ')

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
        example = [(inputs, indices), convert_target(tokens[0])]
    else:
        example = [input,convert_target(tokens[0])]
    if extra:
        example += extra
    return example

def libsvm_load(filename,convert_non_digit_features=float,convert_target=str,sparse=True,input_size=None):
    """
    Reads a LIBSVM file and returns the list of all examples (data)
    and metadata information.

    In general, each example in the list is a two items list ``[input, target]`` where

    * if ``sparse`` is True, ``input`` is a pair (values, indices) of two vectors 
      (vector of values and of indices). Indices start at 1;
    * if ``sparse`` is False, ``input`` is a 1D array such that its elements
      at the positions given by indices-1 are set to the associated values, and the
      other elemnents are 0;
    * ``target`` is a string corresponding to the target to predict.

    If a ``feature:value`` pair in the file is such that ``feature`` is not an integer, 
    ``value`` will be converted to the desired format using option
    ``convert_non_digit_features``. This option must be a callable function
    taking 2 string arguments, and will be called as follows: ::

       output = convert_non_digit_features(feature_str,value_str)

    where ``feature_str`` and ``value_str`` are ``feature`` and ``value`` in string format.
    Its output will be appended to the list of the given example.

    The input_size can be given by the user. Otherwise, will try to figure
    it out from the file (won't work if the file format is sparse and some of the
    last features are all 0!).

    **Defined metadata:**

    * 'targets'
    * 'input_size'

    """

    stream = open(os.path.expanduser(filename))
    data = []
    metadata = {}
    targets = set()
    if input_size is None:
        given_input_size = None
        input_size = 0
    else:
        given_input_size = input_size

    for line in stream:
        example = libsvm_load_line(line,convert_non_digit_features,convert_target,True)
        max_non_zero_feature = max(example[0][1])
        if (given_input_size is None) and (max_non_zero_feature > input_size):
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
    """
    Pickles object ``p`` and saves it to file ``filename``.
    """
    f=file(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def load(filename): 
    """
    Loads pickled object in file ``filename``.
    """
    f=file(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y

def gsave(p, filename):
    """
    Same as ``save(p,filname)``, but saves into a gzipped file.
    """
    f=gfile(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def gload(filename):
    """
    Same as ``load(filname)``, but loads from a gzipped file.
    """    
    f=gfile(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y

