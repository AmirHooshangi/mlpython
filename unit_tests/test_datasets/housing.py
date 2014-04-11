import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_housingloadToMemoryTrue():
    try:
        dataset_store.download('housing')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py housing True')
        dataset_store.delete('housing')
        dataset_store.download('housing')
    except:
        assert False

def test_housingloadToMemoryFalse():
    try:
        dataset_store.download('housing')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py housing False')
        dataset_store.delete('housing')
        dataset_store.download('housing')
    except:
        assert False

