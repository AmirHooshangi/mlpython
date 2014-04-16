import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mnist_background_randomloadToMemoryTrue():
    try:
        dataset_store.download('mnist_background_random')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_background_random True')
        dataset_store.delete('mnist_background_random')
    except:
        assert False

def test_mnist_background_randomloadToMemoryFalse():
    try:
        dataset_store.download('mnist_background_random')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_background_random False')
        dataset_store.delete('mnist_background_random')
    except:
        assert False

