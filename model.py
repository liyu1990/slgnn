from __future__ import print_function

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import warnings
import sys
from encoders import LayerEncoder
from aggregators import LayerAggregator

warnings.filterwarnings("ignore")


class LinkClassifierMLP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden1, hidden2, output = int(dim // 2), int(dim // 4), 1  # final_dim * 4 => final_dim * 2 => final_dim => 1
        activation = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden1),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden1, hidden2),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden2, output)
        )

    def forward(self, x):
        res = self.layers(x)
        res = torch.clamp(res, -1e10, 1e10)
        return torch.sigmoid(res)


class SLGNN(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_feats,
                 features,
                 train_pos=None,
                 train_neg=None,
                 train_pos_dir=None,
                 train_neg_dir=None,
                 test_pos_dir=None,
                 test_neg_dir=None,
                 cuda_available=False,
                 alpha=0.2,
                 node_dropout=0.5,
                 att_dropout=0.5,
                 nheads=4,
                 dim_hidden=64,
                 ):
        super(SLGNN, self).__init__()
        self.num_nodes = num_nodes
        self.train_pos, self.train_neg = train_pos, train_neg
        self.train_pos_dir, self.train_neg_dir = train_pos_dir, train_neg_dir
        self.test_pos_dir, self.test_neg_dir = test_pos_dir, test_neg_dir
        self.cuda_available = cuda_available
        # -----------------------------------------------
        dim_in_layer1, dim_out_layer1, nheads1 = num_feats, dim_hidden, nheads
        dim_in_layer2, dim_out_layer2, nheads2 = dim_out_layer1 * nheads1, dim_hidden, nheads
        dim_out_final = dim_out_layer2
        # -----------------------------------------------
        if cuda_available:
            features.cuda()
        # -----------------------------------------------
        self.agg1 = LayerAggregator(1, features, cuda=cuda_available, dim_in=dim_in_layer1, dim_out=dim_out_layer1,
                                    nheads=nheads1, alpha=alpha, node_dropout=node_dropout, att_dropout=att_dropout)
        self.enc1 = LayerEncoder(1, train_pos, train_neg, self.agg1, base_model=None, last_layer=False)
        self.agg2 = LayerAggregator(2, lambda nodes: self.enc1(nodes), cuda=cuda_available, dim_in=dim_in_layer2, dim_out=dim_out_layer2,
                                    nheads=nheads2, alpha=alpha, node_dropout=node_dropout, att_dropout=att_dropout)
        self.enc = LayerEncoder(2, train_pos, train_neg, self.agg2, base_model=self.enc1, last_layer=True)
        # -----------------------------------------------
        self.nllloss_unbalance = nn.NLLLoss()
        self.link_rep_dimensions = dim_out_final * 4
        self.mlp = LinkClassifierMLP(self.link_rep_dimensions)

    def forward(self, nodes):
        return self.enc(nodes)

    def link_sign_scores(self, link_representations):
        return self.mlp(link_representations)

    @staticmethod
    def generate_link_representation(reps, src, dst):
        src_emb, dst_emb = reps[src], reps[dst]
        link_rep = torch.cat((src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb), dim=1)
        return link_rep

    def loss(self):
        all_nodes_list = list(range(self.num_nodes))
        node_representations = self.forward(all_nodes_list)

        train_i_index = torch.cat((self.train_pos[0], self.train_neg[0]), dim=0)
        train_j_index = torch.cat((self.train_pos[1], self.train_neg[1]), dim=0)
        ys_float = torch.LongTensor([1] * self.train_pos.shape[1] + [0] * self.train_neg.shape[1])

        if self.cuda:
            ys_float = ys_float.cuda()

        # end-to-end link sign classifier using mlp
        link_representations = self.generate_link_representation(node_representations, train_i_index, train_j_index)
        positive_prob = self.link_sign_scores(link_representations)
        negative_prob = 1 - positive_prob
        probs = torch.cat((negative_prob, positive_prob), dim=1)
        probs = torch.clamp(probs, 1e-10, 1)
        probs_log = torch.log(probs)
        loss_mlp = self.nllloss_unbalance(probs_log, ys_float)
        return loss_mlp

    @staticmethod
    def evaluation(test_y, pred_p, name=""):
        pred = np.argmax(pred_p, axis=1)
        test_y = np.array(test_y)
        f1_macro = metrics.f1_score(test_y, pred, average='macro')
        f1_micro = metrics.f1_score(test_y, pred, average='micro')
        f1_weighted = metrics.f1_score(test_y, pred, average='weighted')
        f1_binary = metrics.f1_score(test_y, pred, average='binary')
        auc_prob = metrics.roc_auc_score(test_y, pred_p[:, 1])
        auc_label = metrics.roc_auc_score(test_y, pred)

        print(metrics.confusion_matrix(test_y, pred))
        print(name,
              'f1_mi', f1_micro,
              'f1_ma', f1_macro,
              'f1_wt', f1_weighted,
              'f1_bi', f1_binary,
              'auc_p', auc_prob,
              'auc_l', auc_label,
              )
        sys.stdout.flush()
        return f1_micro, f1_macro, f1_weighted, f1_binary, auc_prob, auc_label

    def test_mlp(self, reps):
        test_i_index = torch.cat((self.test_pos_dir[0], self.test_neg_dir[0]), dim=0).detach().numpy().tolist()
        test_j_index = torch.cat((self.test_pos_dir[1], self.test_neg_dir[1]), dim=0).detach().numpy().tolist()
        test_y = np.array([1] * self.test_pos_dir.shape[1] + [0] * self.test_neg_dir.shape[1])
        # making directed link sign prediction
        test_x = self.generate_link_representation(reps, test_i_index, test_j_index)
        pred_p = self.link_sign_scores(test_x).detach().cpu().numpy()
        pred_p = np.concatenate((1 - pred_p, pred_p), axis=1)
        mlp_dir = self.evaluation(test_y, pred_p, 'mlp_dir')
        # """
        # making undirected link sign prediction
        test_x_inv = self.generate_link_representation(reps, test_j_index, test_i_index)
        pred_p_inv = self.link_sign_scores(test_x_inv).detach().cpu().numpy()
        pred_p_inv = np.concatenate((1 - pred_p_inv, pred_p_inv), axis=1)
        #
        pred_p_all = np.concatenate(([pred_p, pred_p_inv]), axis=0)
        test_y_all = np.concatenate((test_y, test_y))
        mlp_dou = self.evaluation(test_y_all, pred_p_all, 'mlp_dou')
        return mlp_dir, mlp_dou

    def test_func(self, last_epoch=False, path=None):
        all_nodes_list = list(range(self.num_nodes))
        node_representations = self.forward(all_nodes_list)

        metrics_mlp_dir, metrics_mlp_dou = self.test_mlp(node_representations)

        if last_epoch and path is not None:
            if self.cuda:
                node_representations = node_representations.detach().cpu().numpy()
            else:
                node_representations = node_representations.detach().numpy()
            np.savetxt(path, node_representations)

        return metrics_mlp_dir, metrics_mlp_dou
