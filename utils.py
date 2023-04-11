import argparse
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Normalizer
import pickle
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.n_row = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.n_row + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    @staticmethod
    def forward(indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_undirected_networks(file_name, undirected=True):
    links = {}
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        for line in fp:
            line = line.strip()
            if line == "" or "#" in line:
                continue
            rater, rated, sign = [int(val) for val in line.split()]
            assert (sign != 0)
            sign = 1 if sign > 0 else -1

            if not undirected:
                edge1 = (rater, rated)
                if edge1 not in links:
                    links[edge1] = sign
                elif links[edge1] == sign:
                    pass
                else:
                    links[edge1] = -1
                continue

            edge1, edge2 = (rater, rated), (rated, rater)
            if edge1 not in links:
                links[edge1], links[edge2] = sign, sign
            elif links[edge1] == sign:
                pass
            else:
                links[edge1], links[edge2] = -1, -1

    adj_lists_pos, adj_lists_neg = defaultdict(set), defaultdict(set)
    num_edges_pos, num_edges_neg = 0, 0
    for (i, j), s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
            num_edges_pos += 1
        else:
            adj_lists_neg[i].add(j)
            num_edges_neg += 1
    num_edges_pos /= 2
    num_edges_neg /= 2
    return n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg


def load_sparse_adjacency(file_name, undirected=True):
    n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg = load_undirected_networks(file_name, undirected)
    adj_spr_pos_row, adj_spr_pos_col = [], []
    for i in range(n):
        for j in adj_lists_pos[i]:
            adj_spr_pos_row.append(i)
            adj_spr_pos_col.append(j)
    adj_sps_pos = torch.LongTensor([adj_spr_pos_row, adj_spr_pos_col])

    adj_sps_neg_row, adj_sps_neg_col = [], []
    for i in range(n):
        for j in adj_lists_neg[i]:
            adj_sps_neg_row.append(i)
            adj_sps_neg_col.append(j)
    adj_sps_neg = torch.LongTensor([adj_sps_neg_row, adj_sps_neg_col])

    return n, [num_edges_pos, num_edges_neg], adj_sps_pos, adj_sps_neg


def feature_normalization(feature, _type="standard"):
    if _type == "standard":
        return StandardScaler().fit_transform(feature)
    elif _type == "norm_l2":
        return Normalizer().fit_transform(feature)
    else:
        return feature


def read_in_feature_data(feature_train, features_dims):
    print("loading features ... ")
    feat_data = pickle.load(open(feature_train, "rb"))
    print("load tsvd node representation...", end="\t")
    if features_dims is not None:
        feat_data = feat_data[:, :features_dims]
    num_nodes, num_feats = feat_data.shape
    feat_data = feature_normalization(feat_data, "standard")
    print(feat_data.shape)
    return num_feats, feat_data


def load_data(args):
    net_train, feature_train, net_test, features_dims = \
        args['net_train'], args['features_train'], args['net_test'], args['feature_dim']

    num_nodes, num_edges, adj_sps_pos, adj_sps_neg = load_sparse_adjacency(net_train, undirected=True)
    _, _, train_pos_dir, train_neg_dir = load_sparse_adjacency(net_train, undirected=False)
    _, _, test_pos_dir, test_neg_dir = load_sparse_adjacency(net_test, undirected=False)
    num_feats, feat_data = read_in_feature_data(feature_train, features_dims)
    return num_nodes, num_edges, adj_sps_pos, adj_sps_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, feat_data


def calculate_class_weights(num_pos, num_neg):
    num_edges = num_pos + num_neg
    w_pos_neg = 1
    w_pos = round(w_pos_neg * num_neg / num_edges, 2)
    w_neg = round(w_pos_neg - w_pos, 2)
    return w_pos, w_neg
