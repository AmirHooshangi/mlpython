import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_mnist_background_imagesloadToMemoryTrue():
    try:
        dataset_store.download('mnist_background_images')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_background_images True')
        dataset_store.delete('mnist_background_images')
    except:
        assert False

def test_mnist_background_imagesloadToMemoryFalse():
    try:
        dataset_store.download('mnist_background_images')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py mnist_background_images False')
        dataset_store.delete('mnist_background_images')
    except:
        assert False

