from __future__ import print_function
import torch.nn as nn


class LayerEncoder(nn.Module):
    def __init__(self, _id, train_pos, train_neg, aggregator, base_model=None, last_layer=False):
        super(LayerEncoder, self).__init__()
        self.id = _id
        self.last_layer = last_layer
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model

    def forward(self, nodes):
        if self.last_layer:
            return self.aggregator.forward(nodes, self.train_pos, self.train_neg, self.last_layer)
        else:
            return self.aggregator.forward(nodes, self.train_pos, self.train_neg, self.last_layer)
