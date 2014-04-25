import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_adultloadToMemoryTrue():
    try:
        dataset_store.download('adult')
        utGenerator.run_test('adult', True)
        dataset_store.delete('adult')
    except:
        assert False

def test_adultloadToMemoryFalse():
    try:
        dataset_store.download('adult')
        utGenerator.run_test('adult', False)
        dataset_store.delete('adult')
    except:
        assert False

