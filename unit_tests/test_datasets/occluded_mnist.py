import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_occluded_mnistloadToMemoryTrue():
    try:
        dataset_store.download('occluded_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py occluded_mnist True')
        dataset_store.delete('occluded_mnist')
    except:
        assert False

def test_occluded_mnistloadToMemoryFalse():
    try:
        dataset_store.download('occluded_mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py occluded_mnist False')
        dataset_store.delete('occluded_mnist')
    except:
        assert False

