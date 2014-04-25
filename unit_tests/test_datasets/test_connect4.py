import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_connect4loadToMemoryTrue():
    try:
        dataset_store.download('connect4')
        utGenerator.run_test('connect4', True)
        dataset_store.delete('connect4')
    except:
        assert False

def test_connect4loadToMemoryFalse():
    try:
        dataset_store.download('connect4')
        utGenerator.run_test('connect4', False)
        dataset_store.delete('connect4')
    except:
        assert False

