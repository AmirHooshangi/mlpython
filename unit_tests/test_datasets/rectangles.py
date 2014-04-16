import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_rectanglesloadToMemoryTrue():
    try:
        dataset_store.download('rectangles')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rectangles True')
        dataset_store.delete('rectangles')
    except:
        assert False

def test_rectanglesloadToMemoryFalse():
    try:
        dataset_store.download('rectangles')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rectangles False')
        dataset_store.delete('rectangles')
    except:
        assert False

