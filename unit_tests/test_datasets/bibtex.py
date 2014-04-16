import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_bibtexloadToMemoryTrue():
    try:
        dataset_store.download('bibtex')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py bibtex True')
        dataset_store.delete('bibtex')
    except:
        assert False

def test_bibtexloadToMemoryFalse():
    try:
        dataset_store.download('bibtex')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py bibtex False')
        dataset_store.delete('bibtex')
    except:
        assert False

