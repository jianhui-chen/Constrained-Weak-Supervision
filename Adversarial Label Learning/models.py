# Note for ALL model class –– inherit from sklean base.py?

# Need to consider if we want to create an abstract base class


class ALL():
    """
    Adversarial Label Learning Classifier

    """

    def __init__(self, logging=False):
        self.logging = logging      # might want to rename this to verbose
        

class Baseline():
    """
    Baseline Classifier
    Need to add more on its functionality. 
    """

    def __init__(self, logging=False):
        self.logging = logging