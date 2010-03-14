class MLProblem:
    """
    Base class for machine learning problems
    
    The machine learning problem consists simply in 
    an iterator over elements in the data. The metadata 
    is explicitly defined by the user.
    """

    def __init__(self, data=None, metadata={}):
        self.data = data
        self.metadata = {}
        self.metadata.update(metadata)

    def __iter__(self):
        for example in self.data:
            yield example

    def __len__(self):
        if not 'length' not in self.metadata:
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
        self.data = data
        self.metadata = {}
        self.metadata.update(metadata)
        self.subset = subset

    def __iter__(self):
        id = 0
        for example in self.data:
            if id in self.subset:
               yield example
            id += 1

    def __len__(self):
        return len(self.subset)

    def setup(self):
        pass

    def apply_on(self, new_data, new_metadata={}):
        new_problem = SubsetProblem(new_data,new_metadata,subset)
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
    
