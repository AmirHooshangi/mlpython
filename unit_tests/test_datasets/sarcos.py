import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_sarcosloadToMemoryTrue():
    try:
        dataset_store.download('sarcos')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py sarcos True')
        dataset_store.delete('sarcos')
        dataset_store.download('sarcos')
    except:
        assert False

def test_sarcosloadToMemoryFalse():
    try:
        dataset_store.download('sarcos')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py sarcos False')
        dataset_store.delete('sarcos')
        dataset_store.download('sarcos')
    except:
        assert False

