import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mnist_rotatedloadToMemoryTrue():
    try:
        dataset_store.download('mnist_rotated')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_rotated True')
        dataset_store.delete('mnist_rotated')
    except:
        assert False

def test_mnist_rotatedloadToMemoryFalse():
    try:
        dataset_store.download('mnist_rotated')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_rotated False')
        dataset_store.delete('mnist_rotated')
    except:
        assert False

