import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('abalone')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'abalone'
        assert False

def test_abaloneloadToMemoryTrue():
    utGenerator.run_test('abalone', True)

def test_abaloneloadToMemoryFalse():
    utGenerator.run_test('abalone', False)
    print 'test2'

def tearDown():
    dataset_store.delete('abalone')
    print 'teardown'
