import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_cadataloadToMemoryTrue():
    try:
        dataset_store.download('cadata')
        utGenerator.run_test('cadata', True)
        dataset_store.delete('cadata')
    except:
        assert False

def test_cadataloadToMemoryFalse():
    try:
        dataset_store.download('cadata')
        utGenerator.run_test('cadata', False)
        dataset_store.delete('cadata')
    except:
        assert False

