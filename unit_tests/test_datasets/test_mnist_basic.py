import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mnist_basicloadToMemoryTrue():
    try:
        dataset_store.download('mnist_basic')
        utGenerator.run_test('mnist_basic', True)
        dataset_store.delete('mnist_basic')
    except:
        assert False

def test_mnist_basicloadToMemoryFalse():
    try:
        dataset_store.download('mnist_basic')
        utGenerator.run_test('mnist_basic', False)
        dataset_store.delete('mnist_basic')
    except:
        assert False

