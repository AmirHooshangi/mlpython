class RankingProblem(MLProblem):
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
            tot_input += [input]
            tot_target += [target]
            if last_query is not None: # Is not first example
                if last_query is not query: # Yield ranking example if query changed
                    yield (tot_input,tot_target,last_query)
                    tot_input = []
                    tot_target = []
            last_query = query

        if tot_input: # Output last ranking example
            yield (tot_input,tot_target,last_query)

    def __len__(self):
        return self.metadata['n_queries']
