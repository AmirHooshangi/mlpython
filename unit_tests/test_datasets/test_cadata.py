import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('cadata')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'cadata'
        assert False

def test_cadataloadToMemoryTrue():
    utGenerator.run_test('cadata', True)

def test_cadataloadToMemoryFalse():
    utGenerator.run_test('cadata', False)
    print 'test2'

def tearDown():
    dataset_store.delete('cadata')
    print 'teardown'
