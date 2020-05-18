"""
This file defines the core research contribution   
"""

from research_seed.base_module import BaseModule
from research_seed.OneDCNN.modules import OneDimCNNPytorch


class OneDCNN(BaseModule):

    def __init__(self, hparams):
        super(OneDCNN, self).__init__(hparams)

    def init_model(self, input_shape, num_classes):
        self.net = OneDimCNNPytorch(input_shape, num_classes)


