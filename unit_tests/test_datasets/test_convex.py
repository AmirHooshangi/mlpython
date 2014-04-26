import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('convex')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'convex'
        assert False

def test_convexloadToMemoryTrue():
    utGenerator.run_test('convex', True)

def test_convexloadToMemoryFalse():
    utGenerator.run_test('convex', False)
    print 'test2'

def tearDown():
    dataset_store.delete('convex')
    print 'teardown'
