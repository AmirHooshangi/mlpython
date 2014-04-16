import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_nipsloadToMemoryTrue():
    try:
        dataset_store.download('nips')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py nips True')
        dataset_store.delete('nips')
    except:
        assert False

def test_nipsloadToMemoryFalse():
    try:
        dataset_store.download('nips')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py nips False')
        dataset_store.delete('nips')
    except:
        assert False

