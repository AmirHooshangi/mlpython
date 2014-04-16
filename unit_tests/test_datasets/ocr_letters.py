import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_ocr_lettersloadToMemoryTrue():
    try:
        dataset_store.download('ocr_letters')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py ocr_letters True')
        dataset_store.delete('ocr_letters')
    except:
        assert False

def test_ocr_lettersloadToMemoryFalse():
    try:
        dataset_store.download('ocr_letters')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py ocr_letters False')
        dataset_store.delete('ocr_letters')
    except:
        assert False

