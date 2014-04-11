import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mnist_basicloadToMemoryTrue():
    try:
        dataset_store.download('mnist_basic')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_basic True')
        dataset_store.delete('mnist_basic')
        dataset_store.download('mnist_basic')
    except:
        assert False

def test_mnist_basicloadToMemoryFalse():
    try:
        dataset_store.download('mnist_basic')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_basic False')
        dataset_store.delete('mnist_basic')
        dataset_store.download('mnist_basic')
    except:
        assert False

