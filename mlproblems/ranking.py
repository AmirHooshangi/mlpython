import generic as mlpb

class RankingProblem(mlpb.MLProblem):
    """
    Generates a ranking problem from data and metadata.

    The data should be an iterator input/target/query pairs, where the
    target is a relevance score. When grouping query data together,
    is it assumed that examples from the same query are next to each
    other in the data (e.g. there aren't examples from the same query
    at the beginning and end of the data).

    The ranking examples become triplets (inputs_for_query,targets_for_query,query)
    where inputs_for_query and target_for_query are the lists of inputs and targets
    for a given query.

    Required metadata:
    - 'n_queries'

    """

    def __iter__(self):
        tot_input = []
        tot_target = []
        last_query = None

        for input,target,query in self.data:
            if last_query != None: # Is not first example
                if last_query != query: # Yield ranking example if query changed
                    yield (tot_input,tot_target,last_query)
                    tot_input = []
                    tot_target = []
            tot_input += [input]
            tot_target += [target]
            last_query = query

        if tot_input: # Output last ranking example
            yield (tot_input,tot_target,last_query)

    def __len__(self):
        return self.metadata['n_queries']

class RankingToClassificationProblem(mlpb.MLProblem):
    """
    Generates a classification problem from a ranking problem.

    Option 'merge_document_and_query' (a function with 2 arguments)
    is used to generate inputs for the classification problem.
    
    The list of possible scores must be provided as a metadata.
    IMPORTANT: the scores should be ordered from less to more relevant
    in the list. This list will be used to generate the set of targets
    and 'class_to_id' mapping.

    Required_metadata:
    - 'length': number of document/query pairs (if not present, will figure it out, but might be slow)
    - 'scores': list of possible scores, ordered from less relevant to more relevant

    Defined metadata:
    - 'targets'
    - 'class_to_id'

    """

    def __init__(self, data=None, metadata={},merge_document_and_query=None):
        mlpb.MLProblem.__init__(self,data,metadata)
        self.merge_document_and_query = merge_document_and_query

    def __iter__(self):
        for inputs,targets,query in self.data:
            for input,target in zip(inputs,targets):
                if target in self.class_to_id:
                    yield self.merge_document_and_query(input,query),self.class_to_id[target]
                else:
                    yield self.merge_document_and_query(input,query),None # For unlabeled data

    def __len__(self):
        if 'length' not in self.metadata:
            length = 0
            for inputs,targets,query in self.data:
                length += min(len(inputs),len(targets))
            self.metadata['length'] = length
        return self.metadata['length']

    def setup(self):
        # Creating class (string) to id (integer) mapping
        self.class_to_id = {}
        current_id = 0
        for score in self.metadata['scores']:
            self.class_to_id[score] = current_id
            current_id += 1
        self.metadata['class_to_id'] = self.class_to_id
        self.metadata['targets'] = set(self.metadata['scores'])

    def apply_on(self, new_data, new_metadata=None):
        new_problem = RankingToClassificationProblem(new_data,new_metadata)
        new_problem.metadata['class_to_id'] = self.metadata['class_to_id']
        new_problem.metadata['targets'] = self.metadata['targets']
        new_problem.class_to_id = self.class_to_id
        new_problem.merge_document_and_query = self.merge_document_and_query
        return new_problem
