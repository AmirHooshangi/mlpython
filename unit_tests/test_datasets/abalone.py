import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_abaloneloadToMemoryTrue():
    try:
        dataset_store.download('abalone')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py abalone True')
        dataset_store.delete('abalone')
    except:
        assert False

def test_abaloneloadToMemoryFalse():
    try:
        dataset_store.download('abalone')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py abalone False')
        dataset_store.delete('abalone')
    except:
        assert False

