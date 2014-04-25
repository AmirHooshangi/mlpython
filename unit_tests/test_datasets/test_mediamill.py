import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mediamillloadToMemoryTrue():
    try:
        dataset_store.download('mediamill')
        utGenerator.run_test('mediamill', True)
        dataset_store.delete('mediamill')
    except:
        assert False

def test_mediamillloadToMemoryFalse():
    try:
        dataset_store.download('mediamill')
        utGenerator.run_test('mediamill', False)
        dataset_store.delete('mediamill')
    except:
        assert False

