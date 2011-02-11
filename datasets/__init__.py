"""
The ``datasets`` package provides a common framework for downloading
and loading datasets. It is perfect for someone who wishes to
experiment with a Leaner and wants quick access to many arbitrary datasets.

The package include a module for each currently supported dataset. The
module docstring should give a reference of work that produced the dataset
or that used this particular version of the dataset.

The package also has a ``datasets.store`` module implements simple functions
to obtain MLProblems from those datasets.

The modules currently included are:

* ``datasets.store``:             provides functions for obtaining MLProblems from the supported datasets.
* ``datasets.adult``:             Adult dataset module.
* ``datasets.binarized_mnist``:   binarized version of MNIST module.
* ``datasets.connect4``:          Connect-4 dataset module.
* ``datasets.dna``:               DNA dataset module.
* ``datasets.mnist``:             MNIST dataset module.
* ``datasets.mushrooms``:         Mushrooms dataset module.
* ``datasets.newsgroups``:        20-newsgroup dataset module.
* ``datasets.nips``:              NIPS dataset module.
* ``datasets.ocr_letters``:       OCR letters dataset module.
* ``datasets.rcv1``:              RCV1 dataset module.
* ``datasets.web``:               Web dataset module.

"""
