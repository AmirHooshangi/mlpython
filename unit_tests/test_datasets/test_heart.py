import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_heartloadToMemoryTrue():
    try:
        dataset_store.download('heart')
        utGenerator.run_test('heart', True)
        dataset_store.delete('heart')
    except:
        assert False

def test_heartloadToMemoryFalse():
    try:
        dataset_store.download('heart')
        utGenerator.run_test('heart', False)
        dataset_store.delete('heart')
    except:
        assert False

