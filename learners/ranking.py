from generic import Learner
import numpy as np
import mlpython.mlproblems.classification as mlpb

# The training set for these models should be an iterator over 
# triplets (input,target,query), where input is a list of
# document representations and target is a list of associated 
# relevance scores for the given query


class RankingFromClassifier(Learner):
    """ 
    A ranking model based on a classifier.
 
    This learner trains a given classifier to 
    predict the target relevance associated to each
    document/query pairs found in the training set.
    Option 'merge_document_and_query' should be a 
    callable function that takes two arguments (the 
    input document and the query) and outputs a 
    merged representation for the pair which will
    be fed to the classifier.

    Options:
    - 'classifier'
    - 'merge_document_and_query'

    Required metadata:
    - 'targets'

    """
    def __init__(   self,
                    classifier,
                    merge_document_and_query,
                    ):
        self.stage = 0
        self.classifier = classifier
        self.merge_document_and_query=merge_document_and_query

    def train(self,trainset):
        """
        Trains the calssifier on the merged documents and queries.
        Each call to train increments self.stage by 1.
        """

        if self.stage == 0:
            self.classifier_trainset = mlpb.RankingToClassificationProblem(trainset.data,
                                                                           trainset.metadata,
                                                                           self.merge_document_and_query)

        # Training classifier
        self.classifier.train(self.classifier_trainset)
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.classifier.forget()
        self.classifier_trainset=None

    def use(self,dataset):
        """
        Outputs a list corresponding to an ordering of the
        documents, from most relevant to least. 

        For example, ordering [1,3,0,2] means that the 
        second document is most relevant, then the fourth, 
        then the first and finally the third.

        Inspired from http://learningtorankchallenge.yahoo.com/instructions.php
        """
        
        cdataset = self.classifier_dataset.apply_on(dataset.data,dataset.metadata)
        coutputs = self.classifier.use(cdataset)
        offset = 0
        outputs = []
        for inputs,targets,query in dataset:
            preds = coutputs[offset:(offset+len(inputs)),0].ravel()
            preds *= -1
            order = np.argsort(preds)
            outputs += [order]
            offset += len(inputs)
        return outputs

    def err_and_ndcg(self,output,target,k=10):
        """
        Computes the ERR and NDCG score 
        (taken from here: http://learningtorankchallenge.yahoo.com/evaluate.py.txt)
        """

        err = 0.
        ndcg = 0.
        l = [int(x)+1 for x in target]
        r = [int(x)+1 for x in output]
        nd = len(target) # Number of documents
        assert len(output)==nd, 'Expected %d ranks, but got %d.'%(nd,len(r))
        
        gains = [-1]*nd # The first element is the gain of the first document in the predicted ranking
        assert max(r)<=nd, 'Ranks larger than number of documents (%d).'%(nd)
        for j in range(nd):
          gains[r[j]-1] = (2**l[j]-1.0)/16
        assert min(gains)>=0, 'Not all ranks present.'
        
        p = 1.0
        for j in range(nd):
            r = gains[j]
            err += p*r/(j+1.0)
            p *= 1-r
        
        dcg = sum([g/log(j+2) for (j,g) in enumerate(gains[:k])])
        gains.sort()
        gains = gains[::-1]
        ideal_dcg = sum([g/log(j+2) for (j,g) in enumerate(gains[:k])])
        if ideal_dcg:
            ndcg += dcg / ideal_dcg
        else:
            ndcg += 0.5
            
        return (err,ndcg)

    def test(self,dataset):
        """
        Outputs the document ordering and the associated ERR and NDCG scores.
        """
        outputs = self.use(dataset)
        assert len(outputs) == len(dataset)
        costs = np.zeros((len(dataset),2))
        for output,cost,example in zip(outputs,costs,dataset):
            cost[0],cost[1] = self.err_and_ndcg(output,example[1])

        return outputs,costs
