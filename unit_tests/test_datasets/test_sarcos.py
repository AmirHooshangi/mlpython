import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_sarcosloadToMemoryTrue():
    try:
        dataset_store.download('sarcos')
        utGenerator.run_test('sarcos', True)
        dataset_store.delete('sarcos')
    except:
        assert False

def test_sarcosloadToMemoryFalse():
    try:
        dataset_store.download('sarcos')
        utGenerator.run_test('sarcos', False)
        dataset_store.delete('sarcos')
    except:
        assert False

