import argparse
import numpy as np
import random
import sys
import torch
import warnings
import datetime
import torch.nn as nn

from utils import str2bool, load_data, calculate_class_weights
from model import SLGNN

warnings.filterwarnings("ignore")

# ================================================================================= #

parser = argparse.ArgumentParser("Implementation for SLGNN framework")
# input configuration
parser.add_argument('--cuda_available', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)  # -1 for cpu;
parser.add_argument('--net_train', type=str, required=True)
parser.add_argument('--features_train', type=str, required=True)
parser.add_argument('--net_test', type=str, required=False, default=None)
parser.add_argument('--feature_dim', type=int, default=64)

# training details
parser.add_argument('--epoches', type=int, default=2000)  # bitcoinAlpha, bitcoinOtc: 2000; Slashdot, Epinions: 4000
parser.add_argument('--interval', type=int, default=10)  # bitcoinAlpha, bitcoinOtc: 10; Slashdot, Epinions: 200

# saving paths
parser.add_argument('--modify_input_features', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--model_path', type=str, default="model_{}.pkl".format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))
parser.add_argument('--embedding_path', type=str, default="embedding_{}.pkl".format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))

# model parameters
parser.add_argument('--seed', type=int, default=2020)  # random.randint(0, 1e10), we fix the temporary seed as 2020 in the released code.
parser.add_argument('--lr', type=float, default=0.01)  # 0.01
parser.add_argument('--regularize', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.2)  # negative_slope for LeakyReLU
parser.add_argument('--node_dropout', type=float, default=0.5)  # 0.5
parser.add_argument('--att_dropout', type=float, default=0.5)  # 0.5 for bitcoinAlpha, bitcoinOtc, and Slashdot; 0.4 for Epinions
parser.add_argument('--nheads', type=int, default=4)  # 4 for  bitcoinAlpha, Slashdot and Epinions; 2 for bitcoinOtc

parameters = parser.parse_args()
print(parameters)
args = {}
for arg in vars(parameters):
    args[arg] = getattr(parameters, arg)
# ================================================================================= #


rnd_seed = args['seed']
np.random.seed(rnd_seed)
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)

cuda = args['cuda_available']
if args['cuda_device'] == -1:
    cuda = False
if cuda:
    # args['cuda_device'] = torch.cuda.current_device()
    print("Using {} CUDA!!!".format(args['cuda_device']))
    torch.cuda.set_device(args['cuda_device'])
    torch.cuda.manual_seed(rnd_seed)
else:
    print("Using CPU!!!")

num_nodes, num_edges, train_pos, train_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, input_features = load_data(args)
args['class_weights'] = calculate_class_weights(num_edges[0], num_edges[1])

node_features = nn.Embedding(num_nodes, num_feats)
if args['modify_input_features']:
    node_features.weight = nn.Parameter(torch.FloatTensor(input_features), requires_grad=True)
else:
    node_features.weight = nn.Parameter(torch.FloatTensor(input_features), requires_grad=False)

##################################################################


slgnn = SLGNN(num_nodes=num_nodes,
              num_feats=num_feats,
              features=node_features,
              train_pos=train_pos,
              train_neg=train_neg,
              train_pos_dir=train_pos_dir,
              train_neg_dir=train_neg_dir,
              test_pos_dir=test_pos_dir,
              test_neg_dir=test_neg_dir,
              cuda_available=cuda,
              alpha=args['alpha'],
              node_dropout=args['node_dropout'],
              att_dropout=args['att_dropout'],
              nheads=args['nheads'],
              )

if cuda:
    slgnn.cuda()

optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, slgnn.parameters()), lr=args['lr'], weight_decay=args['regularize'])

interval = args['interval']
total_epoches = args['epoches']
minimal_loss, minimal_batch = 1e9, 0
stop_count = 0

for batch in range(total_epoches):
    slgnn.train()
    # ----------------------------------- #
    optimizer.zero_grad()
    loss = slgnn.loss()

    if loss < minimal_loss:
        minimal_loss, minimal_batch = loss, batch
        torch.save(slgnn.state_dict(), args["model_path"])
        stop_count = 0
    else:
        stop_count += 1

    loss.backward()
    optimizer.step()

    print('batch {} loss: {} \t early: {}'.format(batch, loss, stop_count))
    # ----------------------------------- #
    if (batch + 1) % interval == 0 or batch == total_epoches - 1:
        slgnn.eval()
        optimizer.zero_grad()
        _, _ = slgnn.test_func()
    sys.stdout.flush()

print("Training done...", "\n", "-" * 50)
print('Loading model at {}th epoch'.format(minimal_batch))
slgnn.load_state_dict(torch.load(args["model_path"]))
slgnn.eval()
optimizer.zero_grad()
print("Saving node representations at {} epoch!!!".format(minimal_batch))
_, _ = slgnn.test_func(last_epoch=True, path=args['embedding_path'])
