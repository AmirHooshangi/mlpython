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
Module ``datasets.toronto_faces`` gives access to the Toronto Face Dataset.

"""

import scipy.io
import os
import numpy as np

def load(dir_path,fold_id=0,small_images=False,binary_images=False,aligned_images=False,no_rotations=False):
    """
    Loads the Toronto Face Dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The data has been separated into 5 folds, and option ``fold_id`` 
    (from 0 to 4, default is 0) determines which fold is given.

    Each input is a 100x100 image (uint8 values pixels). A smaller (48x48)
    version is also available (set option ``small_images`` to True). A binary
    version of those small images is available too (set option ``binary_images``
    to True). An aligned version of the non-binary images is also available
    (set option ``aligned_images`` to True). An alignement that isn't using
    rotations is also available for 100x100 images (set option no_rotations to 
    True).

    The targets are expression IDs. There are 7 expresions, with IDs from 0 to 6.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'class_to_id'``
    * ``'length'``

    **TODO:**

    * add a ``'unlabeled'`` key to the mapping, giving unlabeled
      data on which unsupervised learning can be done.
    * add possibility to include example meta-information
      (identity IDs, original dataset of origin, etc.).

    """
    dir_path = os.path.expanduser(dir_path)
    if small_images:
        input_size = (48,48)
        if binary_images and aligned_images:
            images = scipy.io.loadmat(os.path.join(dir_path,'data_labeled_48x48_bin.mat'))['data_images'] 

        if binary_images:
            images = scipy.io.loadmat(os.path.join(dir_path,'data_images_aligned_binary_in_folds_48x48.mat'))['data_images'] 
        else:
            if aligned_images:
                if no_rotations:
                    raise ValueError('aligned version without rotations of the 48x48 image dataset is not available')
                else:
                    images = scipy.io.loadmat(os.path.join(dir_path,'data_images_aligned_in_folds_48x48.mat'))['data_images']
            else:
                images = scipy.io.loadmat(os.path.join(dir_path,'data_images_in_folds_48x48.mat'))['data_images']
    else:
        input_size = (100,100)
        if binary_images:
            raise ValueError('binary version of the 100x100 image dataset is not available')
        if aligned_images:
            if no_rotations:
                images = scipy.io.loadmat(os.path.join(dir_path,'data_images_shiftaligned_in_folds_100x100.mat'))['data_images']                
            else:
                raise ValueError('aligned version with rotations of the 100x100 image dataset is not available')
        else:
            images = scipy.io.loadmat(os.path.join(dir_path,'data_images_100x100_in_folds.mat'))['data_images']

    expressions = scipy.io.loadmat(os.path.join(dir_path,'data_info.mat'),struct_as_record=True)['data_info'][0,0]['expressions'].ravel()
    fold = scipy.io.loadmat(os.path.join(dir_path,'data_folds.mat'))['data_folds'][:,fold_id].ravel()

    is_in_fold = np.nonzero(fold>0)
    # Getting train,valid,test images
    train_images = images[:,:,0,np.nonzero(fold[is_in_fold]==1)]
    valid_images = images[:,:,0,np.nonzero(fold[is_in_fold]==2)]
    test_images = images[:,:,0,np.nonzero(fold[is_in_fold]==3)]

    # Getting train,valid,test targets (and scaling them from 0 to 6)
    train_targets = expressions[np.nonzero(fold==1)]-1
    valid_targets = expressions[np.nonzero(fold==2)]-1
    test_targets = expressions[np.nonzero(fold==3)]-1


    # Create iterator over train,valid,test sets
    train,valid,test = [
        zip([set_images[:,:,0,i] for i in xrange(set_images.shape[3])],  # Inputs
            set_targets)                                                 # Targets
        for set_images,set_targets in [(train_images,train_targets),(valid_images,valid_targets),(test_images,test_targets)]]

    # Get metadata
    n_train = train_images.shape[3]
    n_valid = valid_images.shape[3]
    n_test = test_images.shape[3]
    lengths = [l for dummy,dummy,dummy,l in [train_images.shape,valid_images.shape,test_images.shape]]
    
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                        'targets':set(range(7)),
                                        'class_to_id':dict([(id,id) for id in range(7)]),
                                        'length':l} for l in lengths]

    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Gives information about how to obtain this dataset (``dir_path`` is ignored).
    """

    print 'Ask Josh Susskind (http://aclab.ca/users/josh/) for the data.'
 
