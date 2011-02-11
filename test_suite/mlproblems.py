import mlpython.mlproblems.generic as mlpbgen
import mlpython.mlproblems.classification as mlpbclass

raw_data = zip(range(6),['A','A','B','C','A','B'])
metadata = {'length':6,'targets':['A','B','C'],'input_size':1}

def features(example,metadata):
    return ((example[0],example[0]),example[1])

pb1 = mlpbgen.MLProblem(raw_data, metadata)
print 'pb1:'
for example in pb1:
    print example
print 'metadata:',pb1.metadata
print ''

pb2 = mlpbgen.SubsetProblem(pb1,subset=set([1,3,5]))
print 'pb2:'
for example in pb2:
    print example
print 'metadata:',pb2.metadata
print ''

pb3 = mlpbgen.MergedProblem([pb2,pb1])
print 'pb3:'
for example in pb3:
    print example
print 'metadata:',pb3.metadata
print ''

pb4 = mlpbgen.PreprocessedProblem(pb3,metadata={'input_size':2},preprocess=features)
print 'pb4:'
for example in pb4:
    print example
print 'metadata:',pb4.metadata
print ''

pb5 = mlpbclass.ClassificationProblem(pb4)
print 'pb5:'
for example in pb5:
    print example
print 'metadata:',pb5.metadata
print ''

pb6 = mlpbclass.ClassSubsetProblem(pb5,subset=set(['A','C']))
print 'pb6 (final):'
for example in pb6:
    print example
print 'metadata:',pb6.metadata
print ''

pb7 = mlpbgen.SubsetFieldsProblem(pb6,fields=[0,0,1])
print 'pb7 (final):'
for example in pb7:
    print example
print 'metadata:',pb7.metadata
print ''

print 'What is expected:'
final_data = zip([(1,1),(3,3),(0,0),(1,1),(3,3),(4,4),(5,5)],[(1,1),(3,3),(0,0),(1,1),(3,3),(4,4),(5,5)],[0,1,0,0,1,0])
for example in final_data:
    print example
print ''

raw_data2 = zip(range(6,10),['C','B','A','C'])
metadata2 = {'length':4,'targets':['A','B','C'],'input_size':1}

pbtest = pb7.apply_on(raw_data2,metadata2)
print 'pbtest (final):'
for example in pbtest:
    print example
print 'metadata:',pbtest.metadata
print ''

print 'What is expected:'
final_data = zip([(6,6),(8,8),(9,9)],[(6,6),(8,8),(9,9)],[1,0,1])
for example in final_data:
    print example
print ''

