import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_letor_mq2008loadToMemoryTrue():
    try:
        dataset_store.download('letor_mq2008')
        utGenerator.run_test('letor_mq2008', True)
        dataset_store.delete('letor_mq2008')
    except:
        assert False

def test_letor_mq2008loadToMemoryFalse():
    try:
        dataset_store.download('letor_mq2008')
        utGenerator.run_test('letor_mq2008', False)
        dataset_store.delete('letor_mq2008')
    except:
        assert False

