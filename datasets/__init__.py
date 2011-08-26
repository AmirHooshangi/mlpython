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
