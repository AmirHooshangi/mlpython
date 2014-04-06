import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_corrupted_ocr_lettersloadToMemoryTrue():
    try:
        dataset_store.download('corrupted_ocr_letters')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corrupted_ocr_letters True')
        dataset_store.delete('corrupted_ocr_letters')
        dataset_store.download('corrupted_ocr_letters')
    except:
        assert False

def test_corrupted_ocr_lettersloadToMemoryFalse():
    try:
        dataset_store.download('corrupted_ocr_letters')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py corrupted_ocr_letters False')
        dataset_store.delete('corrupted_ocr_letters')
        dataset_store.download('corrupted_ocr_letters')
    except:
        assert False

