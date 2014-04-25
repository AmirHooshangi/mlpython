import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_nipsloadToMemoryTrue():
    try:
        dataset_store.download('nips')
        utGenerator.run_test('nips', True)
        dataset_store.delete('nips')
    except:
        assert False

def test_nipsloadToMemoryFalse():
    try:
        dataset_store.download('nips')
        utGenerator.run_test('nips', False)
        dataset_store.delete('nips')
    except:
        assert False

