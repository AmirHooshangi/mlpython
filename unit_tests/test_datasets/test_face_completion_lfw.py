import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_face_completion_lfwloadToMemoryTrue():
    try:
        dataset_store.download('face_completion_lfw')
        utGenerator.run_test('face_completion_lfw', True)
        dataset_store.delete('face_completion_lfw')
    except:
        assert False

def test_face_completion_lfwloadToMemoryFalse():
    try:
        dataset_store.download('face_completion_lfw')
        utGenerator.run_test('face_completion_lfw', False)
        dataset_store.delete('face_completion_lfw')
    except:
        assert False

