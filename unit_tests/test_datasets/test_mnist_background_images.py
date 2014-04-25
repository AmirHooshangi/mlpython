import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mnist_background_imagesloadToMemoryTrue():
    try:
        dataset_store.download('mnist_background_images')
        utGenerator.run_test('mnist_background_images', True)
        dataset_store.delete('mnist_background_images')
    except:
        assert False

def test_mnist_background_imagesloadToMemoryFalse():
    try:
        dataset_store.download('mnist_background_images')
        utGenerator.run_test('mnist_background_images', False)
        dataset_store.delete('mnist_background_images')
    except:
        assert False

