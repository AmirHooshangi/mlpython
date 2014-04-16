import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
def test_letor_mq2007loadToMemoryTrue():
    try:
        dataset_store.download('letor_mq2007')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py letor_mq2007 True')
        dataset_store.delete('letor_mq2007')
    except:
        assert False

def test_letor_mq2007loadToMemoryFalse():
    try:
        dataset_store.download('letor_mq2007')
        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py letor_mq2007 False')
        dataset_store.delete('letor_mq2007')
    except:
        assert False

