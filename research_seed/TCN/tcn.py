from torch import nn as nn

from research_seed.base_module import BaseModule
from research_seed.TCN.modules import TCN

class TemporalConvolutionalNetwork(BaseModule):

    def __init__(self, hparams):
        super(TemporalConvolutionalNetwork, self).__init__(hparams)

    def init_model(self, input_shape, num_classes):
        nhid = 25
        levels = 8
        channel_sizes = [nhid] * levels
        kernel_size = 7
        dropout = 0.05

        self.net = TCN(input_shape[0], num_classes, channel_sizes, kernel_size=kernel_size,
                                           dropout=dropout)
