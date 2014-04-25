import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_bibtexloadToMemoryTrue():
    try:
        dataset_store.download('bibtex')
        utGenerator.run_test('bibtex', True)
        dataset_store.delete('bibtex')
    except:
        assert False

def test_bibtexloadToMemoryFalse():
    try:
        dataset_store.download('bibtex')
        utGenerator.run_test('bibtex', False)
        dataset_store.delete('bibtex')
    except:
        assert False

