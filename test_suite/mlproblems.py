# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

import mlpython.mlproblems.generic as mlpbgen
import mlpython.mlproblems.classification as mlpbclass

raw_data = zip(range(6),['A','A','B','C','A','B'])
metadata = {'length':6,'targets':['A','B','C'],'input_size':1}

def features(example,metadata):
    metadata['input_size'] = 2
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

pb4 = mlpbgen.PreprocessedProblem(pb3,preprocess=features)
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

