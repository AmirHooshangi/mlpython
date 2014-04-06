import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_face_completion_lfwloadToMemoryTrue():
    try:
        dataset_store.download('face_completion_lfw')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py face_completion_lfw True')
        dataset_store.delete('face_completion_lfw')
        dataset_store.download('face_completion_lfw')
    except:
        assert False

def test_face_completion_lfwloadToMemoryFalse():
    try:
        dataset_store.download('face_completion_lfw')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py face_completion_lfw False')
        dataset_store.delete('face_completion_lfw')
        dataset_store.download('face_completion_lfw')
    except:
        assert False

