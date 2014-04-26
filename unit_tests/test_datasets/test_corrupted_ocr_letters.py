import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('corrupted_ocr_letters')
        print 'setup'
    except:
        print 'Could not download the dataset : ', 'corrupted_ocr_letters'
        assert False

def test_corrupted_ocr_lettersloadToMemoryTrue():
    utGenerator.run_test('corrupted_ocr_letters', True)

def test_corrupted_ocr_lettersloadToMemoryFalse():
    utGenerator.run_test('corrupted_ocr_letters', False)
    print 'test2'

def tearDown():
    dataset_store.delete('corrupted_ocr_letters')
    print 'teardown'
