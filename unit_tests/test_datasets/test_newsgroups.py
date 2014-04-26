import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('newsgroups')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'newsgroups'
        assert False

def test_newsgroupsloadToMemoryTrue():
    utGenerator.run_test('newsgroups', True)

def test_newsgroupsloadToMemoryFalse():
    utGenerator.run_test('newsgroups', False)
    print 'test2'

def tearDown():
    dataset_store.delete('newsgroups')
    print 'teardown'
