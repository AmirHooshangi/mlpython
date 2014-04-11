import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mturkloadToMemoryTrue():
    try:
        dataset_store.download('mturk')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mturk True')
        dataset_store.delete('mturk')
        dataset_store.download('mturk')
    except:
        assert False

def test_mturkloadToMemoryFalse():
    try:
        dataset_store.download('mturk')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mturk False')
        dataset_store.delete('mturk')
        dataset_store.download('mturk')
    except:
        assert False

