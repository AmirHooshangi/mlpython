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
        return len(self.data)

    def setup(self):
        pass

    def apply_on(self, new_data, new_metadata={}):
        new_problem = MLProblem(new_data,new_metadata)
        return new_problem
        
