exp_lists = ["Random", "Five", "Two", "Three", "Four"]

sample_line = "#!/usr/bin/env bash\npython3  /nfs/project/libo_i/Boosting/tools/train_net_step.py --dataset coco2017 " \
              "--cfg /nfs/project/libo_i/Boosting/configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml --use_tfboard " \
              "--bs 8 --nw 4 --iter_size 4 --set FPN.FPN_DISTRIBUTE True FPN.{}_DISTRIBUTE True > " \
              "faster_rcnn_R50_FPN_{}.txt"
file_name_line = "faster_rcnn_R50_FPN_{}.sh"

for item in exp_lists:
	with open(file_name_line.format(item), 'w') as fp:
		fp.write(sample_line.format(item.upper(), item))
