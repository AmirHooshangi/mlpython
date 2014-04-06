import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_heartloadToMemoryTrue():
    try:
        dataset_store.download('heart')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py heart True')
        dataset_store.delete('heart')
        dataset_store.download('heart')
    except:
        assert False

def test_heartloadToMemoryFalse():
    try:
        dataset_store.download('heart')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py heart False')
        dataset_store.delete('heart')
        dataset_store.download('heart')
    except:
        assert False

