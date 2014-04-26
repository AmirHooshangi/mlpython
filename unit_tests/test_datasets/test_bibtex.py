import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('bibtex')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'bibtex'
        assert False

def test_bibtexloadToMemoryTrue():
    utGenerator.run_test('bibtex', True)

def test_bibtexloadToMemoryFalse():
    utGenerator.run_test('bibtex', False)
    print 'test2'

def tearDown():
    dataset_store.delete('bibtex')
    print 'teardown'
