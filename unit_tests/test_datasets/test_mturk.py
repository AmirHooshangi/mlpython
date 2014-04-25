import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mturkloadToMemoryTrue():
    try:
        dataset_store.download('mturk')
        utGenerator.run_test('mturk', True)
        dataset_store.delete('mturk')
    except:
        assert False

def test_mturkloadToMemoryFalse():
    try:
        dataset_store.download('mturk')
        utGenerator.run_test('mturk', False)
        dataset_store.delete('mturk')
    except:
        assert False

