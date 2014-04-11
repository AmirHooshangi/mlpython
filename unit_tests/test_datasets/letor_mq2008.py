import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_letor_mq2008loadToMemoryTrue():
    try:
        dataset_store.download('letor_mq2008')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py letor_mq2008 True')
        dataset_store.delete('letor_mq2008')
        dataset_store.download('letor_mq2008')
    except:
        assert False

def test_letor_mq2008loadToMemoryFalse():
    try:
        dataset_store.download('letor_mq2008')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py letor_mq2008 False')
        dataset_store.delete('letor_mq2008')
        dataset_store.download('letor_mq2008')
    except:
        assert False

