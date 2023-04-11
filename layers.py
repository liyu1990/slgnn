import torch
import torch.nn as nn
from utils import SpecialSpmm


class SpMergeAttentionLayer(nn.Module):
    def __init__(self, dim_in,
                 dim_out,
                 cuda=False,
                 alpha=0.2,
                 num_relations=2,
                 bias=True,
                 node_dropout=0.5,
                 att_dropout=0.5,
                 layer_id=None,
                 basis_att=True,
                 ):
        super(SpMergeAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.cuda = cuda
        self.id = layer_id
        self.basis_att = basis_att

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(1, dim_out)))
            nn.init.xavier_normal_(self.bias.data)
            self.add_bias = True
        else:
            self.add_bias = False

        if self.basis_att:
            self.num_relations = num_relations
            self.num_bases = self.num_relations
            self.basis = nn.Parameter(torch.Tensor(self.num_bases, dim_in, dim_out))
            self.att = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))
            nn.init.xavier_normal_(self.basis.data)
            nn.init.xavier_normal_(self.att.data)
        else:
            self.Wr = nn.Parameter(torch.Tensor(num_relations, dim_in, dim_out))
            nn.init.xavier_normal_(self.Wr.data)

        self.mapping_func = nn.Parameter(torch.zeros(size=(1, dim_out * 2)))
        nn.init.xavier_normal_(self.mapping_func.data)

        self.act_func = nn.LeakyReLU(alpha)
        self.spmm = SpecialSpmm()
        self.dropout_node = nn.Dropout(node_dropout)
        self.dropout_att = nn.Dropout(att_dropout)

    @staticmethod
    def reps_concatenation(src, dst):
        return torch.cat((src, dst), dim=1)

    def forward(self, node_reps, adj_pos, adj_neg, shape=None):
        if shape is None:
            n_row = node_reps.size()[0]
            n_col = n_row
        else:
            n_row, n_col = shape

        num_pos, num_neg = adj_pos.size()[1], adj_neg.size()[1]

        node_reps = self.dropout_node(node_reps)

        if self.basis_att:
            self.Wr = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
            self.Wr = self.Wr.view(self.num_relations, self.dim_in, self.dim_out)

        h_pos, h_neg = torch.mm(node_reps, self.Wr[0]), torch.mm(node_reps, self.Wr[1])

        h_pos_left, h_pos_right = h_pos[adj_pos[0, :], :], h_pos[adj_pos[1, :], :]
        h_neg_left, h_neg_right = h_neg[adj_neg[0, :], :], h_neg[adj_neg[1, :], :]

        sg_rep_pos = self.reps_concatenation(h_pos_left, h_pos_right)
        sg_rep_neg = self.reps_concatenation(h_neg_left, h_neg_right)
        sg_rep_all = torch.cat((sg_rep_pos, sg_rep_neg), dim=0)

        sg_coefficients = torch.sigmoid(self.act_func(self.mapping_func.mm(sg_rep_all.t()).squeeze()))

        tensor_ones_col = torch.ones(size=(n_col, 1))
        if self.cuda:
            tensor_ones_col = tensor_ones_col.cuda()
        adj_all = torch.cat((adj_pos, adj_neg), dim=1)

        edge_row_sum = self.spmm(adj_all, torch.ones_like(sg_coefficients), torch.Size([n_row, n_col]), tensor_ones_col) ** 0.5
        sym_normalize_coefficients = edge_row_sum[adj_all[0]] * edge_row_sum[adj_all[1]]
        sg_coefficients = sg_coefficients.div(sym_normalize_coefficients.squeeze())

        sg_coefficients = self.dropout_att(sg_coefficients)
        h_agg_pos = self.spmm(adj_pos, sg_coefficients[:num_pos], torch.Size([n_row, n_col]), h_pos)
        h_agg_neg = self.spmm(adj_neg, sg_coefficients[-num_neg:], torch.Size([n_row, n_col]), h_neg)

        output = h_agg_pos - h_agg_neg

        if self.add_bias:
            output = output + self.bias
        return output
