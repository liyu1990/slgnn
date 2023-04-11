#!/usr/bin/env bash

cuda_device="0"

data_fold="data"
log_fold="logging"

# -------------------------------------------- #
dataset="bitcoinAlpha"
interval=100
epoches=2000
node_dropout="0.5"
att_dropout="0.5"
head="4"
## -------------------------------------------- #
#dataset="bitcoinOTC"
#interval=100
#epoches=2000
#node_dropout="0.5"
#att_dropout="0.5"
#head="2"
## -------------------------------------------- #
#dataset="slashdot"
#interval=200
#epoches=4000
#node_dropout="0.5"
#att_dropout="0.5"
#head="4"
## -------------------------------------------- #
#dataset="epinions"
#interval=200
#epoches=4000
#node_dropout="0.5"
#att_dropout="0.4"
#head="4"
## -------------------------------------------- #

log_fold="${log_fold}/${dataset}"

if [ ! -d "${log_fold}" ]; then
  mkdir -p "${log_fold}"
fi

indices="0 1 2 3 4"

for idx in ${indices}; do
  output_file_prefix="${log_fold}/${dataset}_${idx}"
  model_file="${output_file_prefix}_model.pkl"
  log_file="${output_file_prefix}_log.log"
  embed_file="${output_file_prefix}_embedding.txt"

  python main.py \
    --nheads ${head} \
    --cuda_device ${cuda_device} \
    --node_dropout ${node_dropout} \
    --att_dropout ${att_dropout} \
    --net_train ${data_fold}/train_test/${dataset}/${dataset}_maxC_train${idx}.edgelist \
    --net_test ${data_fold}/train_test/${dataset}/${dataset}_maxC_test${idx}.edgelist \
    --features_train ${data_fold}/features/${dataset}/${dataset}_maxC_train${idx}_features64_tsvd.pkl \
    --model_path ${model_file} \
    --embedding_path ${embed_file} \
    --interval ${interval} \
    --epoches ${epoches} >$log_file 2>&1 # &
done
