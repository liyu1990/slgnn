import torch
import torch.nn as nn
from layers import SpMergeAttentionLayer


class LayerAggregator(nn.Module):
    def __init__(self, _id, node_reps, cuda=False, dim_in=64, dim_out=64, nheads=4, alpha=0.2, node_dropout=0.5, att_dropout=0.5):
        super(LayerAggregator, self).__init__()
        self.id = _id
        self.node_reps = node_reps
        self.cuda = cuda
        self.act_func = nn.ReLU()

        self.attentions = [
            SpMergeAttentionLayer(dim_in=dim_in, dim_out=dim_out, cuda=cuda, alpha=alpha, node_dropout=node_dropout, att_dropout=att_dropout,
                                  layer_id=_id * nheads + att_id) for att_id in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}_{}'.format(self.id, i), attention)

    def forward(self, nodes, adj_pos, adj_neg, last_layer):
        row_indices_self = [i for i in range(len(nodes))]
        adj_self = torch.LongTensor([row_indices_self, row_indices_self])
        adj_pos2 = torch.cat((adj_pos, adj_self), dim=1)

        if self.cuda:
            adj_pos2 = adj_pos2.cuda()
            adj_neg = adj_neg.cuda()
            node_reps = self.node_reps(torch.LongTensor(row_indices_self).cuda())
        else:
            node_reps = self.node_reps(torch.LongTensor(row_indices_self))

        if len(self.attentions) > 1:
            if last_layer:
                h_hidden = 0
                for idx, att_layer in enumerate(self.attentions):
                    res = att_layer(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self)))
                    h_hidden += res
                h_hidden /= len(self.attentions)
            else:
                h_hidden = torch.cat([
                    att(node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self))) for att in self.attentions], dim=1)
        else:
            h_hidden = self.attentions[0](node_reps, adj_pos2, adj_neg, shape=(len(row_indices_self), len(row_indices_self)))

        h_hidden = self.act_func(h_hidden)
        return h_hidden
