import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('web')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'web'
        assert False

def test_webloadToMemoryTrue():
    utGenerator.run_test('web', True)

def test_webloadToMemoryFalse():
    utGenerator.run_test('web', False)
    print 'test2'

def tearDown():
    dataset_store.delete('web')
    print 'teardown'
