import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_sceneloadToMemoryTrue():
    try:
        dataset_store.download('scene')
        utGenerator.run_test('scene', True)
        dataset_store.delete('scene')
    except:
        assert False

def test_sceneloadToMemoryFalse():
    try:
        dataset_store.download('scene')
        utGenerator.run_test('scene', False)
        dataset_store.delete('scene')
    except:
        assert False

