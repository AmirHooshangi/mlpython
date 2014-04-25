import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mnist_rotatedloadToMemoryTrue():
    try:
        dataset_store.download('mnist_rotated')
        utGenerator.run_test('mnist_rotated', True)
        dataset_store.delete('mnist_rotated')
    except:
        assert False

def test_mnist_rotatedloadToMemoryFalse():
    try:
        dataset_store.download('mnist_rotated')
        utGenerator.run_test('mnist_rotated', False)
        dataset_store.delete('mnist_rotated')
    except:
        assert False

