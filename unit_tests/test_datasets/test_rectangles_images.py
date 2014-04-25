import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_rectangles_imagesloadToMemoryTrue():
    try:
        dataset_store.download('rectangles_images')
        utGenerator.run_test('rectangles_images', True)
        dataset_store.delete('rectangles_images')
    except:
        assert False

def test_rectangles_imagesloadToMemoryFalse():
    try:
        dataset_store.download('rectangles_images')
        utGenerator.run_test('rectangles_images', False)
        dataset_store.delete('rectangles_images')
    except:
        assert False

