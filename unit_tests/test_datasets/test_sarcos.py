import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('sarcos')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'sarcos'
        assert False

def test_sarcosloadToMemoryTrue():
    utGenerator.run_test('sarcos', True)

def test_sarcosloadToMemoryFalse():
    utGenerator.run_test('sarcos', False)
    print 'test2'

def tearDown():
    dataset_store.delete('sarcos')
    print 'teardown'
