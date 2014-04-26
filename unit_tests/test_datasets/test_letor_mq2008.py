import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('letor_mq2008')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'letor_mq2008'
        assert False

def test_letor_mq2008loadToMemoryTrue():
    utGenerator.run_test('letor_mq2008', True)

def test_letor_mq2008loadToMemoryFalse():
    utGenerator.run_test('letor_mq2008', False)
    print 'test2'

def tearDown():
    dataset_store.delete('letor_mq2008')
    print 'teardown'
