import os
import mlpython.datasets.store as dataset_store
import sys

#datasetThatCannotBeDownload = set(['yahoo_ltrc1', 'yahoo_ltrc2'])

datasetToTest = dataset_store.all_names #- datasetThatCannotBeDownload

saveoutput = sys.stdout
for name in datasetToTest:

    output_name = os.path.join(os.getenv('PYTHONPATH'),'mlpython/unit_tests/test_datasets/')
    output_file = open(output_name+ name +'.py','w')
    sys.stdout = output_file

    print "import mlpython.datasets.store as dataset_store"
    print "import os"
    print "from nose.tools import *"
    print "def test_"+name+"loadToMemoryTrue():"
    print "    try:"
    print "        dataset_store.download('"+name+"')"
    print "        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py " + name + " True')"
    print "        dataset_store.delete('"+name+"')"
    print "    except:"
    print "        assert False"
    print
    print "def test_"+name+"loadToMemoryFalse():"
    print "    try:"
    print "        dataset_store.download('"+name+"')"
    print "        os.system(os.environ.get('PYTHONPATH') + '/mlpython/unit_tests/test_datasets/utGenerator.py " + name + " False')"
    print "        dataset_store.delete('"+name+"')"
    print "    except:"
    print "        assert False"
    print
    output_file.close()
sys.stdout = saveoutput
    
