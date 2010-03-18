class MLProblem:
    """
    Base class for machine learning problems
    
    The machine learning problem consists simply in 
    an iterator over elements in the data. The metadata 
    is explicitly defined by the user.

    Options:
    - 'data'
    - 'metadata'

    """

    def __init__(self, data=None, metadata={}):
        self.data = data
        self.metadata = {}
        self.metadata.update(metadata)

    def __iter__(self):
        for example in self.data:
            yield example

    def __len__(self):
        if 'length' not in self.metadata:
            length = len(self.data)
            self.metadata['length'] = length
        return self.metadata['length']

    def setup(self):
        pass

    def apply_on(self, new_data, new_metadata={}):
        new_problem = MLProblem(new_data,new_metadata)
        return new_problem
        
class SubsetProblem(MLProblem):
    """
    Extracts a subset of the examples in a dataset.
    
    The examples that are extracted have their id (example number
    from 0 to size of the dataset-1, as defined by the order
    in which the data iterator yields the examples) in a given subset.

    Options
    - 'subset'

    """

    def __init__(self, data=None, metadata={},subset=set([])):
        MLProblem(self,data,metadata)
        self.subset = subset

    def __iter__(self):
        id = 0
        for example in self.data:
            if id in self.subset:
               yield example
            id += 1

    def __len__(self):
        return len(self.subset)

    def apply_on(self, new_data, new_metadata={}):
        new_problem = SubsetProblem(new_data,new_metadata,self.subset)
        return new_problem

class SubsetFieldsProblem(MLProblem):
    """
    Extracts a subset of the fields in a dataset.
    
    The fields that are selected are given by option
    'fields', a list of indices corresponding to
    the fields to keep. Each example of the new dataset
    will now be a list of those fields, unless 'fields' 
    contains only one index, in which case each example will
    correspond to that field.

    Options
    - 'fields'

    """

    def __init__(self, data=None, metadata={},fields=[0]):
        MLProblem(self,data,metadata)
        self.fields = fields

    def __iter__(self):
        for example in self.data:
            if len(self.subset_fields) == 1:
                yield example[self.fields[0]]
            else:
                yield [example[i] for i in self.fields]

    def apply_on(self, new_data, new_metadata={}):
        new_problem = SubsetFieldsProblem(new_data,new_metadata,self.fields)
        return new_problem

class MergedProblem(MLProblem):
    """
    Merges several datasets together.
    
    Each element of data should itself be an iterator
    over examples.

    """

    def __iter__(self):
        for dataset in self.data:
            for example in dataset:
                yield example

    def __len__(self):
        l = 0
        for dataset in self.data:
            l += len(dataset)
        return l
    
class PreprocessedProblem(MLProblem):
    """
    General class for preprocessing a dataset.

    The examples of this MLProblem is the result
    of applying option 'preprocess' on the examples
    in the original data. Hence, 'preprocess' should
    be a callable function taking one argument (an 
    example from the original data) and returning a 
    preprocessed example.

    IMPORANT: if preprocess changes the size of the inputs, 
    the metadata (i.e. 'input_size') must be changed 
    accordingly by the user.

    Options
    - 'preprocess'

    """

    def __init__(self, data=None, metadata={},preprocess=None):
        MLProblem(self,data,metadata)
        self.preprocess = preprocess

    def __iter__(self):
        for example in self.data:
            yield self.preprocess(example)

    def apply_on(self, new_data, new_metadata={}):
        new_problem = PreprocessedProblem(new_data,new_metadata,self.preprocess)
        return new_problem
