import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mediamillloadToMemoryTrue():
    try:
        dataset_store.download('mediamill')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mediamill True')
        dataset_store.delete('mediamill')
    except:
        assert False

def test_mediamillloadToMemoryFalse():
    try:
        dataset_store.download('mediamill')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mediamill False')
        dataset_store.delete('mediamill')
    except:
        assert False

