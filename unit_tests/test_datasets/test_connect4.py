import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('connect4')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'connect4'
        assert False

def test_connect4loadToMemoryTrue():
    utGenerator.run_test('connect4', True)

def test_connect4loadToMemoryFalse():
    utGenerator.run_test('connect4', False)
    print 'test2'

def tearDown():
    dataset_store.delete('connect4')
    print 'teardown'
