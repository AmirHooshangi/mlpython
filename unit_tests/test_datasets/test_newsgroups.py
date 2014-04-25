import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_newsgroupsloadToMemoryTrue():
    try:
        dataset_store.download('newsgroups')
        utGenerator.run_test('newsgroups', True)
        dataset_store.delete('newsgroups')
    except:
        assert False

def test_newsgroupsloadToMemoryFalse():
    try:
        dataset_store.download('newsgroups')
        utGenerator.run_test('newsgroups', False)
        dataset_store.delete('newsgroups')
    except:
        assert False

