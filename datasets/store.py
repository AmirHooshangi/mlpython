"""
The ``datasets.store`` module provides a unique interface for downloading datasets
and creating MLProblems from those datasets.

So far, it supports the creation of classification and density estimation problems.

It defines the following variables:

* ``datasets.store.all_names``:             set of all dataset names
* ``datasets.store.classification_names``:  set of dataset names for classification
* ``datasets.store.density_names``:         set of dataset names for density estimation
* ``datasets.store.multilabel_names``:      set of dataset names for multilabel classification

It also defines the following functions:

* ``datasets.store.download``:                    downloads a given dataset
* ``datasets.store.get_classification_problem``:  returns a classification MLProblem from some given dataset name
* ``datasets.store.get_density_problem``:         returns a density estimation MLProblem from some given dataset name
* ``datasets.store.get_multilabel_problem``:      returns a multilabel classification MLProblem from some given dataset name

"""

density_names = set(['adult',
                    'binarized_mnist',
                    'connect4',
                    'dna',
                    'mnist',
                    'mushrooms',
                    'nips',
                    'ocr_letters',
                    'rcv1',
                    'web'])

classification_names = set(['adult',
                            'connect4',
                            'dna',
                            'mnist',
                            'mushrooms',
                            'newsgroups',
                            'ocr_letters',
                            'rcv1',
                            'web'])

multilabel_names = set(['mediamill',
                        'scene',
                        'yeast'])

all_names = density_names | classification_names | multilabel_names

def download(name,dataset_dir=None):
    """
    Downloads dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``all_names`` of this module).

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, a subdirectory will be created and the
    dataset will be downloaded there. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in all_names:
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

def get_density_problem(name,dataset_dir=None,load_to_memory=True):
    """
    Creates a density estimation MLProblem from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``density_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in density_names:
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
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_classification_problem(name,dataset_dir=None,load_to_memory=True):
    """
    Creates a classification MLProblem from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``classification_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in classification_names:
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
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_multilabel_problem(name,dataset_dir=None,load_to_memory=True):
    """
    Creates a multilabel classification MLProblem from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``multilabel_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in multilabel_names:
        raise ValueError('dataset '+name+' unknown for multi-label classification learning')
    
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
    trainset = mlpb.MLProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset
