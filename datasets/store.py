density_list = ['adult',
                'binarized_mnist',
                'connect4',
                'dna',
                'mnist',
                'mushrooms',
                'nips',
                'ocr_letters',
                'rcv1',
                'web']

classification_list = ['adult',
                       'connect4',
                       'dna',
                       'mnist',
                       'mushrooms',
                       'newsgroups',
                       'ocr_letters',
                       'rcv1',
                       'web']

all_list = density_list + classification_list

def download(name,dataset_dir=None):
    if name not in all_list:
        raise ValueError('dataset '+name+' unknown')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'
    import os
    if dataset_dir is None:
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    mldataset.obtain(dataset_dir)

def get_density_problem(name,load_to_memory=True,dataset_dir=None):

    if name not in density_list:
        raise ValueError('dataset '+name+' unknown for density learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.generic as mlpb
    if name == 'binarized_mnist' or name == 'nips': 
        trainset = mlpb.MLProblem(train_data,train_metadata)
    else:
        trainset = mlpb.SubsetFieldsProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(valid_data,valid_metadata)

    return trainset,validset,testset

def get_classification_problem(name,load_to_memory=True,dataset_dir=None):

    if name not in classification_list:
        raise ValueError('dataset '+name+' unknown for classification learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.classification as mlpb
    trainset = mlpb.ClassificationProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(valid_data,valid_metadata)

    return trainset,validset,testset
