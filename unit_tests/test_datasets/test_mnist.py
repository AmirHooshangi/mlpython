import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mnist')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'mnist'
        assert False

def test_mnistloadToMemoryTrue():
    utGenerator.run_test('mnist', True)

def test_mnistloadToMemoryFalse():
    utGenerator.run_test('mnist', False)
    print 'test2'

def tearDown():
    dataset_store.delete('mnist')
    print 'teardown'
