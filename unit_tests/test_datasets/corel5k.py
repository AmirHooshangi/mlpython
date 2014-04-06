import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_corel5kloadToMemoryTrue():
    try:
        dataset_store.download('corel5k')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corel5k True')
        dataset_store.delete('corel5k')
        dataset_store.download('corel5k')
    except:
        assert False

def test_corel5kloadToMemoryFalse():
    try:
        dataset_store.download('corel5k')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corel5k False')
        dataset_store.delete('corel5k')
        dataset_store.download('corel5k')
    except:
        assert False

