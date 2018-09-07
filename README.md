# MLPython
Python Machine Learning Framework which aims fast access to many standard datasets (like MNIST, CIFAR, NIPS, yahoo_ltrc1, medical and etc).

This repository is a migration from BitBucket of Hugo Larochelle:
[MLPython](https://bitbucket.org/HugoLarochelle/mlpython) in order to maintain more developer friendly environment, better documentation and pip packaging.

## Installation
For quick installation with pip:
```
pip install mlpython
```
ML python also uses some native codes in C programming language for correct installation please refer to :
[Installation](http://www.dmi.usherb.ca/~larocheh/mlpython/install.html)

# Quick sample 
```
import mlpython.datasets.store as dataset_store
import os

os.environ["MLPYTHON_DATASET_REPO"] = "/home/mylinux_home/directory"
dataset_store.download('mnist')
trainset,validset,testset = dataset_store.get_classification_problem('mnist')

```

## Full Documentation
For the full documentation pleas check here:

[Documentation](http://www.dmi.usherb.ca/~larocheh/mlpython/index.html)

