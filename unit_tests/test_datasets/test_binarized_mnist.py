import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_binarized_mnistloadToMemoryTrue():
    try:
        dataset_store.download('binarized_mnist')
        utGenerator.run_test('binarized_mnist', True)
        dataset_store.delete('binarized_mnist')
    except:
        assert False

def test_binarized_mnistloadToMemoryFalse():
    try:
        dataset_store.download('binarized_mnist')
        utGenerator.run_test('binarized_mnist', False)
        dataset_store.delete('binarized_mnist')
    except:
        assert False

