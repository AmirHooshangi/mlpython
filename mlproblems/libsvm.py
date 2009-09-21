from generic import MLProblem

class LIBSVMClassificationProblem(MLProblem):
    """
    Generates a classification problem from LIBSVM data.
    
    The data must be an iterator and the metadata a dictionary,
    as given by the raw_data.libsvm_data.read() function.
    This class takes that data and metadata and is then an iterator
    over that data, with the new target field now corresponding
    to a class id from 0 to the number of possible targets minus 1.

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
        new_problem = LIBSVMClassificationProblem(new_data,new_metadata)
        new_problem.metadata['class_to_id'] = self.metadata['class_to_id']
        new_problem.class_to_id = self.class_to_id
        return new_problem
        
