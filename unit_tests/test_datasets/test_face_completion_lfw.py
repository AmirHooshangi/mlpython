import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('face_completion_lfw')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'face_completion_lfw'
        assert False

def test_face_completion_lfwloadToMemoryTrue():
    utGenerator.run_test('face_completion_lfw', True)

def test_face_completion_lfwloadToMemoryFalse():
    utGenerator.run_test('face_completion_lfw', False)
    print 'test2'

def tearDown():
    dataset_store.delete('face_completion_lfw')
    print 'teardown'
