import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator
def test_letor_mq2007loadToMemoryTrue():
    try:
        dataset_store.download('letor_mq2007')
        utGenerator.run_test('letor_mq2007', True)
        dataset_store.delete('letor_mq2007')
    except:
        assert False

def test_letor_mq2007loadToMemoryFalse():
    try:
        dataset_store.download('letor_mq2007')
        utGenerator.run_test('letor_mq2007', False)
        dataset_store.delete('letor_mq2007')
    except:
        assert False

