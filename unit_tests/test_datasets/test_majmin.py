import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_majminloadToMemoryTrue():
    try:
        dataset_store.download('majmin')
        utGenerator.run_test('majmin', True)
        dataset_store.delete('majmin')
    except:
        assert False

def test_majminloadToMemoryFalse():
    try:
        dataset_store.download('majmin')
        utGenerator.run_test('majmin', False)
        dataset_store.delete('majmin')
    except:
        assert False

