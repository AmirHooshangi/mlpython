import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('adult')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'adult'
        assert False

def test_adultloadToMemoryTrue():
    utGenerator.run_test('adult', True)

def test_adultloadToMemoryFalse():
    utGenerator.run_test('adult', False)
    print 'test2'

def tearDown():
    dataset_store.delete('adult')
    print 'teardown'
