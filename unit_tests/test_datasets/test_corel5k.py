import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_corel5kloadToMemoryTrue():
    try:
        dataset_store.download('corel5k')
        utGenerator.run_test('corel5k', True)
        dataset_store.delete('corel5k')
    except:
        assert False

def test_corel5kloadToMemoryFalse():
    try:
        dataset_store.download('corel5k')
        utGenerator.run_test('corel5k', False)
        dataset_store.delete('corel5k')
    except:
        assert False

