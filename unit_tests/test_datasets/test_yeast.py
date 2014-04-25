import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_yeastloadToMemoryTrue():
    try:
        dataset_store.download('yeast')
        utGenerator.run_test('yeast', True)
        dataset_store.delete('yeast')
    except:
        assert False

def test_yeastloadToMemoryFalse():
    try:
        dataset_store.download('yeast')
        utGenerator.run_test('yeast', False)
        dataset_store.delete('yeast')
    except:
        assert False

