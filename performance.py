#!/usr/bin/env python
# coding:utf-8


import collections
import os
import numpy as np


def get_file(dataset, idx):
    return "./logging/{0}/{0}_{1}_log.log".format(dataset, idx)


def mean_std(value, key):
    l = value[key]
    avg = round(np.mean(l) * 100, 2)
    std = round(np.std(l) * 100, 2)
    return "{}: {}/{}".format(key, avg, std)


def get_values(line):
    values = line.split()
    d = {}
    for i in range(1, len(values), 2):
        d[values[i]] = float(values[i + 1])
    return d


def statistics(dataset, mode="mlp_dou"):
    print("-" * 10, "# {} - {} #".format(dataset, mode), "-" * 10)
    results = collections.defaultdict(lambda: collections.defaultdict(list))

    for idx in range(5):
        filename = get_file(dataset, idx)
        if not os.path.exists(filename): continue
        mlp_dir, mlp_dou = "", ""
        with open(filename, "r") as fp:
            for line in fp:
                if "mlp_dir" in line:  mlp_dir = line.strip()
                if "mlp_dou" in line:  mlp_dou = line.strip()

        for t in get_values(mlp_dir).items():
            results["mlp_dir"][t[0]].append(t[1])

        for t in get_values(mlp_dou).items():
            results["mlp_dou"][t[0]].append(t[1])

    value = results[mode]
    f1_mi, f1_ma, f1_wt, f1_bi, auc_p, auc_l = mean_std(value, "f1_mi"), mean_std(value, "f1_ma"), mean_std(value, "f1_wt"), \
                                               mean_std(value, "f1_bi"), mean_std(value, "auc_p"), mean_std(value, "auc_l")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(mode, f1_mi, f1_ma, f1_wt, f1_bi, auc_p, auc_l))


print("-" * 100)

if __name__ == '__main__':
    dataset = "bitcoinAlpha"
    # dataset = "bitcoinOTC"
    # dataset = "slashdot"
    # dataset = "epinions"

    # mode = "mlp_dir"
    mode = "mlp_dou"

    statistics(dataset, mode)
