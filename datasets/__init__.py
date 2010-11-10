"""
The ``datasets`` package provides a common framework for downloading
and loading datasets. It is perfect for someone who wishes to
experiment with a Leaner and wants quick access to many arbitrary datasets.

The package include a module for each currently supported dataset. The
module docstring should give a reference of work that used this particular
version of the dataset.

The package also has a ``datasets.store`` module implements simple functions
to obtain MLProblems from those datasets.

The modules currently included are:

* ``datasets.store``:             provides functions for obtaining MLProblems from the supported datasets.
* ``datasets.adult``:             functions for downloading and loading data from the Adult dataset.
* ``datasets.binarized_mnist``:   functions for downloading and loading data from a binarized version of MNIST.
* ``datasets.connect4``:          functions for downloading and loading data from the connect-4 dataset.
* ``datasets.dna``:               functions for downloading and loading data from the dna dataset.
* ``datasets.mnist``:             functions for downloading and loading data from the MNIST dataset.
* ``datasets.mushrooms``:         functions for downloading and loading data from a binarized version of MNIST.
* ``datasets.newsgroups``:        functions for downloading and loading data from the 20newsgroup dataset.
* ``datasets.nips``:              functions for downloading and loading data from the NIPS dataset.
* ``datasets.ocr_letters``:       functions for downloading and loading data from the OCR letters dataset.
* ``datasets.rcv1``:              functions for downloading and loading data from the RCV1 dataset.
* ``datasets.web``:               functions for downloading and loading data from the web dataset.

"""
