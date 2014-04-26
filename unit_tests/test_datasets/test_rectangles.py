import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('rectangles')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'rectangles'
        assert False

def test_rectanglesloadToMemoryTrue():
    utGenerator.run_test('rectangles', True)

def test_rectanglesloadToMemoryFalse():
    utGenerator.run_test('rectangles', False)
    print 'test2'

def tearDown():
    dataset_store.delete('rectangles')
    print 'teardown'
