import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_dnaloadToMemoryTrue():
    try:
        dataset_store.download('dna')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py dna True')
        dataset_store.delete('dna')
    except:
        assert False

def test_dnaloadToMemoryFalse():
    try:
        dataset_store.download('dna')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py dna False')
        dataset_store.delete('dna')
    except:
        assert False

