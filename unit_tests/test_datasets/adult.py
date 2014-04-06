import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_adultloadToMemoryTrue():
    try:
        dataset_store.download('adult')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py adult True')
        dataset_store.delete('adult')
        dataset_store.download('adult')
    except:
        assert False

def test_adultloadToMemoryFalse():
    try:
        dataset_store.download('adult')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py adult False')
        dataset_store.delete('adult')
        dataset_store.download('adult')
    except:
        assert False

