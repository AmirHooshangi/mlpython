import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mushroomsloadToMemoryTrue():
    try:
        dataset_store.download('mushrooms')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mushrooms True')
        dataset_store.delete('mushrooms')
    except:
        assert False

def test_mushroomsloadToMemoryFalse():
    try:
        dataset_store.download('mushrooms')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mushrooms False')
        dataset_store.delete('mushrooms')
    except:
        assert False

