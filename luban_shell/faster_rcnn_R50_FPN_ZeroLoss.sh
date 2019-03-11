#!/usr/bin/env bash
python3  /nfs/project/libo_i/Boosting/tools/train_net_step.py \
        --dataset coco2017 \
        --cfg /nfs/project/libo_i/Boosting/configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml \
        --use_tfboard --bs 8 --nw 4 --iter_size 4 --my_output \
        --output /nfs/project/libo_i/Boosting/ZeroLoss_Target \
        --set RPN.ZEROLOSS True > faster_rcnn_R50_FPN_ZeroLoss.txt