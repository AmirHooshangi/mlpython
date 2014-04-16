import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_rcv1loadToMemoryTrue():
    try:
        dataset_store.download('rcv1')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rcv1 True')
        dataset_store.delete('rcv1')
    except:
        assert False

def test_rcv1loadToMemoryFalse():
    try:
        dataset_store.download('rcv1')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py rcv1 False')
        dataset_store.delete('rcv1')
    except:
        assert False

