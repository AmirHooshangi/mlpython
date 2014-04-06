import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_binarized_mnistloadToMemoryTrue():
    try:
        dataset_store.download('binarized_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py binarized_mnist True')
        dataset_store.delete('binarized_mnist')
        dataset_store.download('binarized_mnist')
    except:
        assert False

def test_binarized_mnistloadToMemoryFalse():
    try:
        dataset_store.download('binarized_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py binarized_mnist False')
        dataset_store.delete('binarized_mnist')
        dataset_store.download('binarized_mnist')
    except:
        assert False

