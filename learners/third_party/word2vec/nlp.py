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

import subprocess
import uuid
import os
import tempfile
import shlex
from mlpython.learners.generic import Learner

class Word2Vec(Learner):
    def __init__(self, 
                 delete_temporary_files= True,
                 order=3,
                 number_of_threads=12,
                 use_continuous_bag_of_words =0,
                 size = 100,
                 display_script_output = False):
        self.is_trained = False
        self.delete_created_files = delete_temporary_files
        self.use_continuous_bag_of_words = use_continuous_bag_of_words
        self.size = size
        #self.smoothing = smoothing
        #self.n_distributed_language_models = n_distributed_language_models
        #self.pruning = pruning
        #self.sentence_boundary = sentence_boundary
        #self.max_dictionary_size = max_dictionary_size
        #self.min_frequency = min_frequency
        self.number_of_threads = number_of_threads
        self.script_output = None
        self.display_script_output = display_script_output 
        #"vectors.bin -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1 ./distance vectors.bin"

    def train(self,trainset):
        
        if self.is_trained :
            return

        self.script_output = None
        if not self.display_script_output:
            self.script_output = open(os.devnull,'w')
       
        unique_id = uuid.uuid4().hex


        tmp_dir = tempfile.gettempdir()+'/word2vectmp_'+unique_id
        if not os.path.exists(tmp_dir) :
            os.makedirs(tmp_dir)

        input_name = os.path.join(tmp_dir,'inputword2vec'+'.tmp')
        output_name = os.path.join(tmp_dir,'vectors.bin')

        input_file = open(input_name,'w')
        tmplist = list()
        #process_path = os.path.join(os.getenv('PYTHONPATH'),'mlpython/learners/third_party/word2vec/trunk/word2vec')
        #print "potato"
        #print process_path
        '''iterat through the files'''
        var = trainset[0]
        for x in var:
            tmplist.append(' '.join(x))
        #print type(var)
        input_file.write(' '.join(tmplist))
        input_file.close()
        #input_file = open(input_name,'r')
        #output_file = open(output_name, 'w')

        # Get the location of the shell script
        process_path = os.getenv('PYTHONPATH') +'/mlpython/learners/third_party/word2vec/trunk/word2vec'#os.path.join(os.getenv('PYTHONPATH'),'mlpython/learners/third_party/word2vec/trunk/word2vec')
        print process_path
        process_call = process_path + ' -train ' + input_name  + ' -output ' + output_name + ' -cbow '+ str(int(self.use_continuous_bag_of_words)) + ' -size ' + str(self.size) + ' -window 5 -negative 0 -hs 1 -sample 1e-3 -threads ' + str(self.number_of_threads) + ' -binary 1 ./distance vectors.bin'
        args = shlex.split(process_call)
        print args

        #output_file = open(output_name,'w')
        subprocess.Popen(args)
        #subprocess.call([process_call],stdout=output_file, stderr=self.script_output)
        #stdin=input_file,

        #if(self.delete_temporary_files):"""


    def use(self, dataset):

        #Load trained vector

    def forget(self):
        #Delete temporary files

    def test(self, dataset):
        raise NotImplementedError


    def __del__(self):
        if self.delete_created_files:
            self.forget()
