import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_housingloadToMemoryTrue():
    try:
        dataset_store.download('housing')
        utGenerator.run_test('housing', True)
        dataset_store.delete('housing')
    except:
        assert False

def test_housingloadToMemoryFalse():
    try:
        dataset_store.download('housing')
        utGenerator.run_test('housing', False)
        dataset_store.delete('housing')
    except:
        assert False

