import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_rectanglesloadToMemoryTrue():
    try:
        dataset_store.download('rectangles')
        utGenerator.run_test('rectangles', True)
        dataset_store.delete('rectangles')
    except:
        assert False

def test_rectanglesloadToMemoryFalse():
    try:
        dataset_store.download('rectangles')
        utGenerator.run_test('rectangles', False)
        dataset_store.delete('rectangles')
    except:
        assert False

