import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_medicalloadToMemoryTrue():
    try:
        dataset_store.download('medical')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py medical True')
        dataset_store.delete('medical')
    except:
        assert False

def test_medicalloadToMemoryFalse():
    try:
        dataset_store.download('medical')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py medical False')
        dataset_store.delete('medical')
    except:
        assert False

