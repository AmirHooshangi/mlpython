import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_cadataloadToMemoryTrue():
    try:
        dataset_store.download('cadata')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py cadata True')
        dataset_store.delete('cadata')
        dataset_store.download('cadata')
    except:
        assert False

def test_cadataloadToMemoryFalse():
    try:
        dataset_store.download('cadata')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py cadata False')
        dataset_store.delete('cadata')
        dataset_store.download('cadata')
    except:
        assert False

