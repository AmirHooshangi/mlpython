import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_connect4loadToMemoryTrue():
    try:
        dataset_store.download('connect4')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py connect4 True')
        dataset_store.delete('connect4')
    except:
        assert False

def test_connect4loadToMemoryFalse():
    try:
        dataset_store.download('connect4')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py connect4 False')
        dataset_store.delete('connect4')
    except:
        assert False

