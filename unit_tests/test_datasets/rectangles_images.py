import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_rectangles_imagesloadToMemoryTrue():
    try:
        dataset_store.download('rectangles_images')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rectangles_images True')
        dataset_store.delete('rectangles_images')
    except:
        assert False

def test_rectangles_imagesloadToMemoryFalse():
    try:
        dataset_store.download('rectangles_images')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rectangles_images False')
        dataset_store.delete('rectangles_images')
    except:
        assert False

