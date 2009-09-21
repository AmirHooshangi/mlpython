from numpy import zeros

def read(filename):
    """
    Reads a LIBSVM file and returns the list of all examples (data) and metadata information.

    Each example in the list is a pair (input, target) where
    - input is also a pair (values, indices) of two vectors (vector of values and of indices)
    - target is a string corresponding to the target to predict

    Inputs at index smaller than 1 are ignored.

    Required metadata: None

    Defined metadata: 
    - 'targets'
    - 'input_size'

    """
    stream = open(filename)
    data = []
    metadata = {}
    targets = set()
    input_size = 0
    for line in stream:
        line = line.strip()
        tokens = line.split()
        targets.add(tokens[0])

        # Remove indices < 1
        n_removed = 0
        for token,i in zip(tokens, range(len(tokens))):
            if token.find(':') >= 0 and int(token[:token.find(':')]) < 1:
                del tokens[i-n_removed]
                n_removed += 1
            
        inputs = zeros((len(tokens)-1))
        indices = zeros((len(tokens)-1),dtype='int')
        for token,i in zip(tokens[1:], range(len(tokens)-1)):
            id_str,input_str = token.split(':')
            indices[i] = int(id_str)
            inputs[i] = float(input_str)
        if (max(indices)+1) > input_size:
            input_size = max(indices)+1
        data += [((inputs, indices), tokens[0])]

    metadata['targets'] = targets
    metadata['input_size'] = input_size
    return data, metadata
        
            
            
