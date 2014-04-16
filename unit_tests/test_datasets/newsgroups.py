import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_newsgroupsloadToMemoryTrue():
    try:
        dataset_store.download('newsgroups')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py newsgroups True')
        dataset_store.delete('newsgroups')
    except:
        assert False

def test_newsgroupsloadToMemoryFalse():
    try:
        dataset_store.download('newsgroups')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py newsgroups False')
        dataset_store.delete('newsgroups')
    except:
        assert False

