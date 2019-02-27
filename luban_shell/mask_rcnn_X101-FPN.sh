#!/usr/bin/env bash
python3  /nfs/project/libo_i/Boosting/tools/train_net_step.py \
        --dataset coco2017 \
        --cfg /nfs/project/libo_i/Boosting/configs/baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml \
        --use_tfboard --bs 8 --iter_size 4 --nw 4