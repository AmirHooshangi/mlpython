import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mnistloadToMemoryTrue():
    try:
        dataset_store.download('mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist True')
        dataset_store.delete('mnist')
    except:
        assert False

def test_mnistloadToMemoryFalse():
    try:
        dataset_store.download('mnist')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist False')
        dataset_store.delete('mnist')
    except:
        assert False

