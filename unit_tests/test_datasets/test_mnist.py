import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mnistloadToMemoryTrue():
    try:
        dataset_store.download('mnist')
        utGenerator.run_test('mnist', True)
        dataset_store.delete('mnist')
    except:
        assert False

def test_mnistloadToMemoryFalse():
    try:
        dataset_store.download('mnist')
        utGenerator.run_test('mnist', False)
        dataset_store.delete('mnist')
    except:
        assert False

