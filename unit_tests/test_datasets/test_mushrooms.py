import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_mushroomsloadToMemoryTrue():
    try:
        dataset_store.download('mushrooms')
        utGenerator.run_test('mushrooms', True)
        dataset_store.delete('mushrooms')
    except:
        assert False

def test_mushroomsloadToMemoryFalse():
    try:
        dataset_store.download('mushrooms')
        utGenerator.run_test('mushrooms', False)
        dataset_store.delete('mushrooms')
    except:
        assert False

