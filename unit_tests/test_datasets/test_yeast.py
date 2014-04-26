import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('yeast')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'yeast'
        assert False

def test_yeastloadToMemoryTrue():
    utGenerator.run_test('yeast', True)

def test_yeastloadToMemoryFalse():
    utGenerator.run_test('yeast', False)
    print 'test2'

def tearDown():
    dataset_store.delete('yeast')
    print 'teardown'
