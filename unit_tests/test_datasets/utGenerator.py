#!/usr/bin/python

import mlpython.datasets.store as dataset_store
import cPickle,sys
import numpy as np
import sys
import os
from pdb import set_trace as dbg
import itertools

def load(f): 
    """
    Loads pickled object in file ``f``.
    """
    return cPickle.load(f)

def recursiveCompare(element1, element2):
    
    if isinstance(element1,list):
        for x,y in itertools.izip_longest(element1, element2):
            recursiveCompare(x,y)
    elif isinstance(element1, np.ndarray):
        assert (element1 == element2).all()
    else:
        assert element1 == element2


def testfirstandlast(myIterator,myFile, numbertotest =10):
    listToTest = list()
    for i in range(numbertotest):
        listToTest.append(myIterator.next())
    for value in listToTest:
        x = load(myFile)
        recursiveCompare(value,x)

    for value in myIterator:
        listToTest.pop(0)
        listToTest.append(value)
    for value in listToTest:
        x = type(load(myFile))

def main(args):
    usage = """Usage: arg1 = dataset name, arg2 = load_to_memory"""
    if len(args) != 3:
        print usage
        return False
    datasetName = args[1]
    load_to_memory = args[2]
    f = file('./unit_tests/test_datasets/pickles/'+datasetName+'.pkl','rb')

    dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + datasetName
    exec 'import mlpython.datasets.'+datasetName+' as mldataset'
    dictionnary = mldataset.load(dataset_dir, load_to_memory)

    train = dictionnary['train']
    valid = dictionnary['valid']
    test = dictionnary['test']

    testfirstandlast(iter(train[0]),f)
    testfirstandlast(iter(valid[0]),f)
    testfirstandlast(iter(test[0]),f)

    for x in train[1]:
        assert x ==load(f)
    for x in valid[1]:
        assert x ==load(f)
    for x in test[1]:
        assert x ==load(f)

    f.close()

if __name__ == "__main__":
    main(sys.argv)
