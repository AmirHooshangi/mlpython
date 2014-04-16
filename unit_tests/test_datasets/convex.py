import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_convexloadToMemoryTrue():
    try:
        dataset_store.download('convex')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py convex True')
        dataset_store.delete('convex')
    except:
        assert False

def test_convexloadToMemoryFalse():
    try:
        dataset_store.download('convex')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py convex False')
        dataset_store.delete('convex')
    except:
        assert False

