from generic import MLProblem

class ClassificationProblem(MLProblem):
    """
    Generates a classification problem from data and metadata.

    The data should be an iterator input/target pairs. 
    The metadata should contain the set of possible classes 
    (field 'targets' in metadata).
    
    Required metadata: 
    - 'targets'

    Defined metadata: 
    - 'class_to_id'

    """

    def __iter__(self):
        for input,target in self.data:
            yield input,self.class_to_id[target]

    def setup(self):
        # Creating class (string) to id (integer) mapping
        self.class_to_id = {}
        current_id = 0
        for target in self.metadata['targets']:
            self.class_to_id[target] = current_id
            current_id += 1
        self.metadata['class_to_id'] = self.class_to_id

    def apply_on(self, new_data, new_metadata=None):
        new_problem = ClassificationProblem(new_data,new_metadata)
        new_problem.metadata['class_to_id'] = self.metadata['class_to_id']
        new_problem.class_to_id = self.class_to_id
        return new_problem

class ClassSubsetProblem(MLProblem):
    """
    Extracts examples in a dataset belonging to some class.
    
    Options
    - 'subset'
    - 'include_class'

    Defined metadata: 
    - 'class_to_id'
    - 'targets'

    """

    def __init__(self, data=None, metadata={},
                 subset=[], # Subset of classes to include
                 include_class=True # Whether to include the class field
                 ):
        self.data = data
        self.metadata = {}
        self.metadata.update(metadata)
        self.subset=subset
        self.include_class = include_class
        self.class_to_id = {}
        self.targets = set([])
        id = 0
        for c in subset:
            self.class_to_id[c] = id
            self.targets.add(c)
            id+=1
        self.metadata['targets'] = self.targets
        self.metadata['class_to_id'] = self.class_to_id

    def __iter__(self):
        for input,target in self.data:
            if target in self.subset:
                if self.include_class:
                    yield input,self.class_to_id[target]
                else:
                    yield input

    def __len__(self):
        if 'class_subset_length' not in self.metadata:
            length = 0
            for example in self:
                length+=1
            self.metadata['class_subset_length'] = length
        return self.metadata['class_subset_length']

    def setup(self):
        pass

    def apply_on(self, new_data, new_metadata=None):
        new_problem = ClassSubsetProblem(new_data,new_metadata,subset=self.subset,
                                         include_class=self.include_class)
        
        new_problem.targets = self.targets
        new_problem.class_to_id = self.class_to_id
        new_problem.metadata['targets'] = self.targets
        new_problem.metadata['class_to_id'] = self.class_to_id
        return new_problem
        
