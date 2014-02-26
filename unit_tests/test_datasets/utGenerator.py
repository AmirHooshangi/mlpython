import mlpython.datasets.store as dataset_store
import cPickle,sys
import numpy as np
import sys

def load(f): 
    """
    Loads pickled object in file ``f``.
    """
    y=cPickle.load(f)
    return y
def main(args):
    usage = """Usage: arg1 = dataset name arg2 = mlproblem to use """
    if len(args) != 3:
        print usage
        return False
    datasetName = args[1]
    problem = args[2]
    f =file('./datasets/'+datasetName+'.pkl','rb')

    func = getattr(dataset_store,'get_'+problem+'_problem')
    train,valid,test = func(datasetName, None,  load_to_memory=True)
    iteratorTrain = iter(train)
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
        assert (value == load(f)).all()

    '''
    use The list as a circular buffer. keep only the 10 lasts elements
    '''
    for value in iteratorTrain:
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
        assert (value == load(f)).all()

    f.close()
    return True

if __name__ == "__main__":
    main(sys.argv)
