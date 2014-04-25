import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_dnaloadToMemoryTrue():
    try:
        dataset_store.download('dna')
        utGenerator.run_test('dna', True)
        dataset_store.delete('dna')
    except:
        assert False

def test_dnaloadToMemoryFalse():
    try:
        dataset_store.download('dna')
        utGenerator.run_test('dna', False)
        dataset_store.delete('dna')
    except:
        assert False

