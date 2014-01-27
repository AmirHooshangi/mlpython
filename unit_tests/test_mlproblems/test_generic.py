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
from mlpython.mlproblems.generic import MLProblem
import numpy as np

class TestMLProblem:


    def test_len(self):
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3}
        self.mlpb = MLProblem(data,metadata)
        assert len(self.mlpb) == 10

    def test_len_metadata(self):
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3, 'length':9}
        self.mlpb = MLProblem(data,metadata)
        assert len(self.mlpb) == 9