import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('occluded_mnist')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'occluded_mnist'
        assert False

def test_occluded_mnistloadToMemoryTrue():
    utGenerator.run_test('occluded_mnist', True)

def test_occluded_mnistloadToMemoryFalse():
    utGenerator.run_test('occluded_mnist', False)
    print 'test2'

def tearDown():
    dataset_store.delete('occluded_mnist')
    print 'teardown'
