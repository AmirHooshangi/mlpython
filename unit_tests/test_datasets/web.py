import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_webloadToMemoryTrue():
    try:
        dataset_store.download('web')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py web True')
        dataset_store.delete('web')
    except:
        assert False

def test_webloadToMemoryFalse():
    try:
        dataset_store.download('web')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py web False')
        dataset_store.delete('web')
    except:
        assert False

