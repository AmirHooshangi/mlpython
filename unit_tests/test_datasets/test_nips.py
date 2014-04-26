import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('nips')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'nips'
        assert False

def test_nipsloadToMemoryTrue():
    utGenerator.run_test('nips', True)

def test_nipsloadToMemoryFalse():
    utGenerator.run_test('nips', False)
    print 'test2'

def tearDown():
    dataset_store.delete('nips')
    print 'teardown'
