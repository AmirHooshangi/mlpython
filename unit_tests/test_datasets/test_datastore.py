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
The ``test_dataset.test_datastore`` module contains unit tests for the DatasetStore that are 
designed for classification problem.

This module contains the following classes:

"""
import mlpython.datasets.store as dataset_store
from nose.tools import *

class TestDatastoreDataset:

	"""@raises(KeyError)
	def test_missing_metadata(self):
		data = np.arange(10).reshape(5,2)
		cpb = ClassificationProblem(data)
"""
	
	def test_StoreClassification(self):
		ClassificationDatasets = dataset_store.classification_names
		knownClassificationDatasets = set(['heart', 'mnist_rotated_background_images', 'dna', 'mushrooms', 'newsgroups', 'connect4', 'mnist_basic', 'rectangles_images', 'mnist_background_random', 'mnist_rotated', 'mnist_background_images', 'web', 'adult', 'rcv1', 'convex', 'rectangles', 'mnist', 'ocr_letters'])
		assert knownClassificationDatasets.issubset(ClassificationDatasets)

	def test_StoreRegression(self):
		RegressionDatasets = dataset_store.regression_names
		knownRegressionDatasets = set(['housing', 'abalone', 'cadata'])
		assert knownRegressionDatasets.issubset(RegressionDatasets)

	#def test_StoreCollaborativeFiltering(self):
	#	CollaborativeFilteringDatasets = dataset_store.collaborative_filtering_name
	#	knownCollaborativeFilteringDatasets = set([])
	#	assert knownCollaborativeFilteringDatasets.issubset(CollaborativeFilteringDatasets)
	####### The variable collaborative_filtering_name is missing

	def test_StoreDistribution(self):
		DistributionDatasets = dataset_store.distribution_names
		knownDistributionDatasets = set(['heart', 'dna', 'web', 'binarized_mnist', 'connect4', 'rcv1', 'mushrooms', 'adult', 'nips', 'mnist', 'ocr_letters'])
		assert knownDistributionDatasets.issubset(DistributionDatasets)

	def test_StoreMultilabel(self):
		MultilabelDatasets = dataset_store.multilabel_names
		knownMultilabelDatasets = set(['yeast', 'corrupted_mnist', 'medical', 'mediamill', 'mturk', 'scene', 'corrupted_ocr_letters', 'bibtex', 'occluded_mnist', 'corel5k', 'majmin'])
		assert knownMultilabelDatasets.issubset(MultilabelDatasets)

	def test_StoreMultiregression(self):
		MultiregressionDatasets = dataset_store.multiregression_names
		knownMultiregressionDatasets = set(['face_completion_lfw', 'sarcos', 'occluded_faces_lfw'])
		assert knownMultiregressionDatasets.issubset(MultiregressionDatasets)

	def test_StoreRanking(self):
		rankingDataset = dataset_store.ranking_names
		knownRankingDatasets = set(['yahoo_ltrc1', 'letor_mq2008', 'yahoo_ltrc2', 'letor_mq2007'])
		assert knownRankingDatasets.issubset(rankingDataset)

	#def test_StoreNlp(self):
	#	NlpDatasets = dataset_store.nlp_names
	#	knownNlpDatasets = set([])
	#	assert knownNlpDatasets.issubset(NlpDatasets)
	####### The variable nlp_names is missing