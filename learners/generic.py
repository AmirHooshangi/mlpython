

class Learner:
    """
    Base class for a learning algorithm.

    This class is meant to standardize the creation of learners.

    """

    #def __init__():
    
    def train(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def forget(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def use(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def test(self):
        raise NotImplementedError("Subclass should have implemented this method.")

