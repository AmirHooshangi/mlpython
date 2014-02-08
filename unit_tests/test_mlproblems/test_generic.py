# Copyright 2014 Hugo Larochelle. All rights reserved.
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

"""
The ``test_mlproblems.test_generic`` module contains unit tests for MLProblems that are not
designed for a specific type of problem.

This module contains the following classes:

* TestMLProblem:              Root test class for machine learning problems.
* TestSubsetProblem:          Tests extraction a subset of examples from a dataset.
* TestSubsetFieldsProblem:    Tests extraction a subset of the fields in a dataset.
* TestMergedProblem:          Tests the merge of several datasets together.
* TestPreprocessedProblem:    Tests the application of an arbitrary preprocessing on a dataset.
* TestSemisupervisedProblem:  Tests if the SemisupervisedProblem class removed the labels of a subset of the examples in a dataset.

"""
from mlpython.mlproblems.generic import *
import numpy as np

class TestMLProblem:


    def test_len(self):
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3}
        mlpb = MLProblem(data,metadata)
        assert len(mlpb) == 10

    def test_len_metadata(self):
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3, 'length':9}
        mlpb = MLProblem(data,metadata)
        assert len(mlpb) == 9
        #This is a lie that should never be done in a non-test context
        #Given length should always be the real length.

    def test_iter(self):
        data = np.arange(30).reshape((10,3))
        mlpb = MLProblem(data)
        line = 0.
        for example in mlpb:
            array = np.array([line, line+1, line+2])
            assert np.array_equal(array, example)
            line += 3

    def test_peak(self):
        data = np.arange(20).reshape((4,5))
        mlpb = MLProblem(data)
        peakLine = np.array([0,1,2,3,4])
        assert np.array_equal(peakLine, mlpb.peak())

    def test_apply_on(self):
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3, 'length':9}
        mlpb = MLProblem(data,metadata)

        new_data = np.arange(20).reshape((4,5))
        new_metadata = {'input_size':5, 'length':4}
        mlpb2 = MLProblem(new_data, new_metadata)

        assert len(mlpb2) == 4
        assert len(mlpb) == 9

        mlpb3 = mlpb.apply_on(mlpb2)
        assert len(mlpb3) == 4
        assert mlpb3.metadata['input_size'] == 5

class TestSubsetProblem:

    def test_len(self):
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,5])
        subpb = SubsetProblem(data,{},True, subset)

        assert len(subpb) == 3

    def test_iter(self):
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        subpb = SubsetProblem(data,{},True, subset)
        
        results = np.array((0,1,2,3,4,5,12,13,14)).reshape(3,3)
        i = 0
        for line in subpb:
            assert np.array_equal(line, results[i])
            i+=1

    def test_apply_on(self):
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        subpb = SubsetProblem(data,{},True, subset)

        new_data = np.arange(10).reshape((5,2))
        fullSized = subpb.apply_on(new_data)
        assert len(fullSized) == 5

    def test_apply_on_parents(self):
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        mlpb = MLProblem(data)

        child = SubsetProblem(mlpb,{},False, subset)
        print len(child)
        assert len(child) == 3

        new_data = np.arange(10).reshape((5,2))
        fullSized = child.apply_on(new_data)
        assert len(fullSized) == 5

class TestSubsetFieldsProblem:

    def test_len(self):
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        assert len(sfpb) == 4

    def test_iter(self):
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        results = results = np.array((0,1,4,5,6,9,10,11,14,15,16,19)).reshape(4,3)
        i=0
        for line in sfpb:
            assert np.array_equal(line, results[i])
            i+=1

    def test_apply_on(self):
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        data = np.arange(21).reshape((3,7))
        mlpb = MLProblem(data)
        newProblem = sfpb.apply_on(mlpb)

        results = results = np.array((0,1,4,7,8,11,14,15,18)).reshape(3,3)
        i=0
        for line in newProblem:
            assert np.array_equal(line, results[i])
            i+=1

class TestMergedProblem:

    def test_len_serial(self):
        data1 = np.arange(20).reshape((4,5))
        data2 = np.arange(30).reshape((3,10))
        data3 = np.arange(40).reshape((20,2))
        data4 = np.arange(50).reshape((5,10))

        mergedpb = MergedProblem([data1,data2,data3,data4],{}, False, True)

        assert len(mergedpb) == 32

    def test_len_not_serial(self):
        data1 = np.arange(20).reshape((4,5))
        data2 = np.arange(30).reshape((3,10))
        data3 = np.arange(40).reshape((20,2))
        data4 = np.arange(50).reshape((5,10))

        mergedpb = MergedProblem([data1,data2,data3,data4],{}, False, False)
        assert len(mergedpb) == 80 #Max length * number of datasets

    def test_iter_serial(self):
        data1 = np.arange(6).reshape((2,3))
        data2 = np.arange(9).reshape((3,3))
        data3 = np.arange(12).reshape((4,3))

        mergedpb = MergedProblem([data1,data2,data3],{}, False, True)

        results = np.array((0,1,2,3,4,5,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,9,10,11)).reshape(9,3)
        i = 0
        for line in mergedpb:
            assert np.array_equal(line,results[i])
            i+=1

    def test_iter_non_serial(self):
        data1 = np.arange(6).reshape((2,3))
        data2 = np.arange(9).reshape((3,3))
        data3 = np.arange(12).reshape((4,3))

        mergedpb = MergedProblem([data1,data2,data3],{}, False, False)

        results = np.array((0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,0,1,2,6,7,8,6,7,8,3,4,5,0,1,2,9,10,11)).reshape(12,3)
        i = 0
        for line in mergedpb:
            assert np.array_equal(line,results[i])
            i+=1