import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_corrupted_mnistloadToMemoryTrue():
    try:
        dataset_store.download('corrupted_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corrupted_mnist True')
        dataset_store.delete('corrupted_mnist')
        dataset_store.download('corrupted_mnist')
    except:
        assert False

def test_corrupted_mnistloadToMemoryFalse():
    try:
        dataset_store.download('corrupted_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corrupted_mnist False')
        dataset_store.delete('corrupted_mnist')
        dataset_store.download('corrupted_mnist')
    except:
        assert False

