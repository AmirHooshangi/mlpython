import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_sceneloadToMemoryTrue():
    try:
        dataset_store.download('scene')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py scene True')
        dataset_store.delete('scene')
        dataset_store.download('scene')
    except:
        assert False

def test_sceneloadToMemoryFalse():
    try:
        dataset_store.download('scene')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py scene False')
        dataset_store.delete('scene')
        dataset_store.download('scene')
    except:
        assert False

