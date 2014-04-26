import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('corel5k')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'corel5k'
        assert False

def test_corel5kloadToMemoryTrue():
    utGenerator.run_test('corel5k', True)

def test_corel5kloadToMemoryFalse():
    utGenerator.run_test('corel5k', False)
    print 'test2'

def tearDown():
    dataset_store.delete('corel5k')
    print 'teardown'
