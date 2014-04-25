import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_medicalloadToMemoryTrue():
    try:
        dataset_store.download('medical')
        utGenerator.run_test('medical', True)
        dataset_store.delete('medical')
    except:
        assert False

def test_medicalloadToMemoryFalse():
    try:
        dataset_store.download('medical')
        utGenerator.run_test('medical', False)
        dataset_store.delete('medical')
    except:
        assert False

