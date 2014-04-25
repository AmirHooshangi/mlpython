import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_corrupted_mnistloadToMemoryTrue():
    try:
        dataset_store.download('corrupted_mnist')
        utGenerator.run_test('corrupted_mnist', True)
        dataset_store.delete('corrupted_mnist')
    except:
        assert False

def test_corrupted_mnistloadToMemoryFalse():
    try:
        dataset_store.download('corrupted_mnist')
        utGenerator.run_test('corrupted_mnist', False)
        dataset_store.delete('corrupted_mnist')
    except:
        assert False

