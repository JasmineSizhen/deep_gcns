#! /bin/bash

for test_set in {8}
do
    for eval_set in {1..10}
    do
        if [[ $test_set -ne $eval_set ]]; 
        then
            python -u main.py --batch_size 128 --gcn_aggr "add" --hidden_channels 128 --lr 0.001 --mlp_layers 1 --num_layers 9 --val_set $eval_set --test_set $test_set --use_gpu --device 1 > gcn.ngs.capping.val$eval_set.test$test_set.log 
        fi
    done
done