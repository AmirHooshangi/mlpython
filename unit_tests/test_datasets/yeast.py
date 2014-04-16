import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_yeastloadToMemoryTrue():
    try:
        dataset_store.download('yeast')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py yeast True')
        dataset_store.delete('yeast')
    except:
        assert False

def test_yeastloadToMemoryFalse():
    try:
        dataset_store.download('yeast')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py yeast False')
        dataset_store.delete('yeast')
    except:
        assert False

