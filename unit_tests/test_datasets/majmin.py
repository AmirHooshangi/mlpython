import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_majminloadToMemoryTrue():
    try:
        dataset_store.download('majmin')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py majmin True')
        dataset_store.delete('majmin')
        dataset_store.download('majmin')
    except:
        assert False

def test_majminloadToMemoryFalse():
    try:
        dataset_store.download('majmin')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py majmin False')
        dataset_store.delete('majmin')
        dataset_store.download('majmin')
    except:
        assert False

