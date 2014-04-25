import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_rcv1loadToMemoryTrue():
    try:
        dataset_store.download('rcv1')
        utGenerator.run_test('rcv1', True)
        dataset_store.delete('rcv1')
    except:
        assert False

def test_rcv1loadToMemoryFalse():
    try:
        dataset_store.download('rcv1')
        utGenerator.run_test('rcv1', False)
        dataset_store.delete('rcv1')
    except:
        assert False

