# coding:utf-8
import json
from pycocotools.coco import COCO
import os
import sys
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description = 'Select part json')
	parser.add_argument('--number', help = 'the number to select')
	parser.add_argument(
		'--start', default = 1, type = int, help = 'start point ot select')
	parser.add_argument('--json_dir',
	                    type = str,
	                    default = '/nfs/project/data/coco/annotations',
	                    help = 'dir of json file')
	parser.add_argument('--json_name',
	                    type = str,
	                    default = 'instances_val2017',
	                    help = 'name of json file')
	
	args = parser.parse_args()
	return args


def select_part(start, end, coco, json_file):
	images = []
	annotations = []
	categories = []
	imgs_key = list(coco.imgs.keys())
	for i in range(start, end):
		img_key = imgs_key[i]
		images.append(coco.imgs[img_key])
		annoIds_to_img_i = coco.getAnnIds(imgIds = img_key)
		for j in annoIds_to_img_i:
			annotations.append(coco.anns[j])
	for k, _ in coco.cats.items():
		categories.append(coco.cats[k])
	
	json_info = {
		'categories':categories,
		'images':images,
		'annotations':annotations
	}
	with open(json_file, 'w') as sjf:
		# cPickle.dump(json_info, sjf)
		json.dump(json_info, sjf)


if __name__ == '__main__':
	args = parse_args()
	
	task = args.json_name
	json_dir = args.json_dir
	json_name = r'{}.json'.format(task)
	json_path = os.path.join(json_dir, json_name)
	# /nfs/project/data/coco/annotations/instances_train2017.json
	coco = COCO(json_path)
	total_cnt = len(coco.imgs)
	start = args.start
	end = start + int(args.number)
	if end >= total_cnt:
		raise ValueError("Should change your range(from {} to {} with total count {}) to select, the end point "
		                 "exceeds the total count of dataset.".format(start, end, total_cnt))
	select_json_name = r'{}_{}_{}.json'.format(task, start, end)
	select_json_path = os.path.join(json_dir, select_json_name)
	select_part(start, end, coco, os.path.join(json_dir, select_json_path))
