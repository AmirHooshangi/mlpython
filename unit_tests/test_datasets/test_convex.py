import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_convexloadToMemoryTrue():
    try:
        dataset_store.download('convex')
        utGenerator.run_test('convex', True)
        dataset_store.delete('convex')
    except:
        assert False

def test_convexloadToMemoryFalse():
    try:
        dataset_store.download('convex')
        utGenerator.run_test('convex', False)
        dataset_store.delete('convex')
    except:
        assert False

