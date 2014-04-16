import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_abaloneloadToMemoryTrue():
    try:
        dataset_store.download('abalone')
        utGenerator.run_test('abalone',True)
        dataset_store.delete('abalone')
    except:
        assert False

def test_abaloneloadToMemoryFalse():
    try:
        dataset_store.download('abalone')
        utGenerator.run_test('abalone',True)
        dataset_store.delete('abalone')
    except:
        assert False

