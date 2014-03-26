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
    #print type(element1)
    #print element1
    #print type(element2)
    #print element2
    
    if isinstance(element1,list):
        for x,y in itertools.izip_longest(element1, element2):
            #print "x: ", x
            #print "y: ", y
            recursiveCompare(x,y)
    elif isinstance(element1, np.ndarray):
    #    print "array!", type(element1)
        assert (element1 == element2).all()
    else:
        assert element1 == element2


def testfirstandlast(myIterator,myFile, numbertotest =10):
    listToTest = list()
    for i in range(numbertotest):
        listToTest.append(myIterator.next())
    for value in listToTest:
        #print value
        x = load(myFile)
        recursiveCompare(value,x)
        #dbg()
        #assert len(value) == len(set(load(myFile)) & set(value))
    for value in myIterator:
        listToTest.pop(0)
        listToTest.append(value)
    for value in listToTest:
        #assert len(value) == len(set(load(myFile)) & set(value))
    #    print type(value)
        x = type(load(myFile))
    #    print cmp(value,x)
    #dbg()

def main(args):
    usage = """Usage: arg1 = dataset name, arg2 = load_to_memory"""
    if len(args) != 3:
        print usage
        return False
    datasetName = args[1]
    load_to_memory = args[2]
    f = file('./datasets/'+datasetName+'.pkl','rb')

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

    '''iteratorTrain = iter(train)
    iteratorValid = iter(valid)
    iteratorTest = iter(test)
    listTrain = list()
    listValid = list()
    listTest = list()

    for i in range(10):
        listTrain.append(iteratorTrain.next())
        listValid.append(iteratorValid.next())
        listTest.append(iteratorTest.next())
    for value in listTrain:
        assert (value == load(f)).all()
    for value in listValid:
        assert (value == load(f)).all()
    for value in listTest:
        assert (value == load(f)).all()'''

    '''
    use The list as a circular buffer. keep only the 10 lasts elements
    '''
    '''for value in iteratorTrain:
        listTrain.pop(0)
        listTrain.append(value)
    for value in iteratorValid:
        listValid.pop(0)
        listValid.append(value)
    for value in iteratorTest:
        listTest.pop(0)
        listTest.append(value)

    for value in listTrain:
        assert (value == load(f)).all()
    for value in listValid:
        assert (value == load(f)).all()
    for value in listTest:
        assert (value == load(f)).all()'''

    f.close()
    #return True

if __name__ == "__main__":
    main(sys.argv)
