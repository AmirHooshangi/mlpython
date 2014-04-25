# Copyright 2014 Frederic Bergeron & Benoit Gauthier. All rights reserved.
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

import os
import mlpython.datasets.store as dataset_store
import sys

datasetThatCannotBeDownload = set(['yahoo_ltrc1', 'yahoo_ltrc2'])

datasetToTest =  dataset_store.all_names - datasetThatCannotBeDownload

saveoutput = sys.stdout
for name in datasetToTest:
    output_name = os.path.join(os.getenv('PYTHONPATH'),'mlpython/unit_tests/test_datasets/')
    output_file = open(output_name + "test_"+ name +'.py','w')
    sys.stdout = output_file

    print "import mlpython.datasets.store as dataset_store"
    print "import os"
    print "from nose.tools import *"
    print "import utGenerator"
    print "def test_"+name+"loadToMemoryTrue():"
    print "    try:"
    print "        dataset_store.download('"+name+"')"
    print "        utGenerator.run_test('" + name + "', True)"
    print "        dataset_store.delete('"+name+"')"
    print "    except:"
    print "        assert False"
    print
    print "def test_"+name+"loadToMemoryFalse():"
    print "    try:"
    print "        dataset_store.download('"+name+"')"
    print "        utGenerator.run_test('" + name + "', False)"
    print "        dataset_store.delete('"+name+"')"
    print "    except:"
    print "        assert False"
    print
    output_file.close()
sys.stdout = saveoutput
    
