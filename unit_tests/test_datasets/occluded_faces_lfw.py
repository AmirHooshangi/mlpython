import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_occluded_faces_lfwloadToMemoryTrue():
    try:
        dataset_store.download('occluded_faces_lfw')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py occluded_faces_lfw True')
        dataset_store.delete('occluded_faces_lfw')
    except:
        assert False

def test_occluded_faces_lfwloadToMemoryFalse():
    try:
        dataset_store.download('occluded_faces_lfw')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py occluded_faces_lfw False')
        dataset_store.delete('occluded_faces_lfw')
    except:
        assert False

