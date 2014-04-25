import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_occluded_mnistloadToMemoryTrue():
    try:
        dataset_store.download('occluded_mnist')
        utGenerator.run_test('occluded_mnist', True)
        dataset_store.delete('occluded_mnist')
    except:
        assert False

def test_occluded_mnistloadToMemoryFalse():
    try:
        dataset_store.download('occluded_mnist')
        utGenerator.run_test('occluded_mnist', False)
        dataset_store.delete('occluded_mnist')
    except:
        assert False

