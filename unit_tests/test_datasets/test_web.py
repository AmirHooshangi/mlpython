import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_webloadToMemoryTrue():
    try:
        dataset_store.download('web')
        utGenerator.run_test('web', True)
        dataset_store.delete('web')
    except:
        assert False

def test_webloadToMemoryFalse():
    try:
        dataset_store.download('web')
        utGenerator.run_test('web', False)
        dataset_store.delete('web')
    except:
        assert False

