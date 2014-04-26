import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('rectangles_images')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'rectangles_images'
        assert False

def test_rectangles_imagesloadToMemoryTrue():
    utGenerator.run_test('rectangles_images', True)

def test_rectangles_imagesloadToMemoryFalse():
    utGenerator.run_test('rectangles_images', False)
    print 'test2'

def tearDown():
    dataset_store.delete('rectangles_images')
    print 'teardown'
