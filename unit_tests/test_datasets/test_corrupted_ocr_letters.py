import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_corrupted_ocr_lettersloadToMemoryTrue():
    try:
        dataset_store.download('corrupted_ocr_letters')
        utGenerator.run_test('corrupted_ocr_letters', True)
        dataset_store.delete('corrupted_ocr_letters')
    except:
        assert False

def test_corrupted_ocr_lettersloadToMemoryFalse():
    try:
        dataset_store.download('corrupted_ocr_letters')
        utGenerator.run_test('corrupted_ocr_letters', False)
        dataset_store.delete('corrupted_ocr_letters')
    except:
        assert False
