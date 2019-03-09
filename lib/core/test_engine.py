# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer
import utils.boxes as box_utils
from six.moves import cPickle as pickle
import os
import json

# Use a non-interactive backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

logger = logging.getLogger(__name__)


def stats_calculator(all_proposals, gt_i):
	iou_mat = box_utils.bbox_overlaps(all_proposals, gt_i)
	max_inds = np.argmax(iou_mat, axis = 1)
	max_element = np.max(iou_mat, axis = 1)
	thrsh_inds = np.where(max_element >= 0)
	max_inds = max_inds[thrsh_inds]
	all_proposals = all_proposals[thrsh_inds]
	
	# IOU(Intersection Over Union)
	max_element = max_element[thrsh_inds]
	
	center_point_distance = []
	iou_over_gt = []
	
	for ind, item in enumerate(all_proposals):
		# item是迭代中的pp, matched_gt是迭代中的ground_truth
		matched_gt_ind = max_inds[int(ind)]
		matched_gt = gt_i[matched_gt_ind]
		matched_gt_width = matched_gt[2] - matched_gt[0] + 1
		matched_gt_height = matched_gt[3] - matched_gt[1] + 1
		
		gt_center_point = (matched_gt_width / 2, matched_gt_height / 2)
		
		pp_width = item[2] - item[0] + 1
		pp_height = item[3] - item[1] + 1
		
		pp_center_point = (pp_width / 2, pp_height / 2)
		
		distance = np.sqrt(np.square(gt_center_point[0] - pp_center_point[0]) + np.square(gt_center_point[1] - pp_center_point[1]))
		
		# DoC: Distance of Centers(normalized)
		dis_width = matched_gt_width / 2 + pp_width / 2
		dis_height = matched_gt_height / 2 + pp_height / 2
		
		distance = distance / np.sqrt(np.square(dis_width) + np.square(dis_height))
		
		# 计算intersect面积
		center_point_distance.append(distance)
		iw = min(item[2], matched_gt[2]) - max(item[0], matched_gt[0]) + 1
		intersect = 0
		if iw > 0:
			ih = min(item[3], matched_gt[3]) - max(item[1], matched_gt[1]) + 1
			if ih > 0:
				intersect = iw * ih
		
		gt_area = matched_gt_height * matched_gt_width
		assert gt_area > 0
		# Intersection Over GT
		iou_over_gt.append(intersect / gt_area)
	
	center_point_distance = np.array(center_point_distance, dtype = np.float32)
	iou_over_gt = np.array(iou_over_gt, dtype = np.float32)
	return max_element, center_point_distance, iou_over_gt


def get_eval_functions():
	# Determine which parent or child function should handle inference
	if cfg.MODEL.RPN_ONLY:
		raise NotImplementedError
	# child_func = generate_rpn_on_range
	# parent_func = generate_rpn_on_dataset
	else:
		# Generic case that handles all network types other than RPN-only nets
		# and RetinaNet
		child_func = test_net
		parent_func = test_net_on_dataset
	
	return parent_func, child_func


def get_inference_dataset(index, is_parent = True):
	assert is_parent or len(cfg.TEST.DATASETS) == 1, \
		'The child inference process can only work on a single dataset'
	
	dataset_name = cfg.TEST.DATASETS[index]
	
	if cfg.TEST.PRECOMPUTED_PROPOSALS:
		assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
			'The child inference process can only work on a single proposal file'
		assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
			'If proposals are used, one proposal file must be specified for ' \
			'each dataset'
		proposal_file = cfg.TEST.PROPOSAL_FILES[index]
	else:
		proposal_file = None
	
	return dataset_name, proposal_file


def run_inference(
		args, ind_range = None,
		multi_gpu_testing = False, gpu_id = 0,
		check_expected_results = False):
	parent_func, child_func = get_eval_functions()
	is_parent = ind_range is None
	
	def result_getter():
		if is_parent:
			# Parent case:
			# In this case we're either running inference on the entire dataset in a
			# single process or (if multi_gpu_testing is True) using this process to
			# launch subprocesses that each run inference on a range of the dataset
			all_results = {}
			for i in range(len(cfg.TEST.DATASETS)):
				dataset_name, proposal_file = get_inference_dataset(i)
				output_dir = args.output_dir
				results = parent_func(
					args,
					dataset_name,
					proposal_file,
					output_dir,
					multi_gpu = multi_gpu_testing
				)
				all_results.update(results)
			
			return all_results
		else:
			# Subprocess child case:
			# In this case test_net was called via subprocess.Popen to execute on a
			# range of inputs on a single dataset
			dataset_name, proposal_file = get_inference_dataset(0, is_parent = False)
			output_dir = args.output_dir
			return child_func(
				args,
				dataset_name,
				proposal_file,
				output_dir,
				ind_range = ind_range,
				gpu_id = gpu_id
			)
	
	all_results = result_getter()
	if check_expected_results and is_parent:
		task_evaluation.check_expected_results(
			all_results,
			atol = cfg.EXPECTED_RESULTS_ATOL,
			rtol = cfg.EXPECTED_RESULTS_RTOL
		)
		task_evaluation.log_copy_paste_friendly_results(all_results)
	
	return all_results


def test_net_on_dataset(
		args,
		dataset_name,
		proposal_file,
		output_dir,
		multi_gpu = False,
		gpu_id = 0):
	"""Run inference on a dataset."""
	dataset = JsonDataset(dataset_name)
	test_timer = Timer()
	test_timer.tic()
	if multi_gpu:
		num_images = len(dataset.get_roidb())
		all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
			args, dataset_name, proposal_file, num_images, output_dir
		)
	else:
		all_boxes, all_segms, all_keyps = test_net(
			args, dataset_name, proposal_file, output_dir, gpu_id = gpu_id
		)
	test_timer.toc()
	logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
	results = task_evaluation.evaluate_all(
		dataset, all_boxes, all_segms, all_keyps, output_dir
	)
	return results


def multi_gpu_test_net_on_dataset(
		args, dataset_name, proposal_file, num_images, output_dir):
	"""Multi-gpu inference on a dataset."""
	binary_dir = envu.get_runtime_dir()
	binary_ext = envu.get_py_bin_ext()
	binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
	assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)
	
	# Pass the target dataset and proposal file (if any) via the command line
	opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
	if proposal_file:
		opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]
	
	# Run inference in parallel in subprocesses
	# Outputs will be a list of outputs from each subprocess, where the output
	# of each subprocess is the dictionary saved by test_net().
	outputs = subprocess_utils.process_in_parallel(
		'detection', num_images, binary, output_dir,
		args.load_ckpt, args.load_detectron, opts
	)
	
	# Collate the results from each subprocess
	all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	for det_data in outputs:
		all_boxes_batch = det_data['all_boxes']
		all_segms_batch = det_data['all_segms']
		all_keyps_batch = det_data['all_keyps']
		for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
			all_boxes[cls_idx] += all_boxes_batch[cls_idx]
			all_segms[cls_idx] += all_segms_batch[cls_idx]
			all_keyps[cls_idx] += all_keyps_batch[cls_idx]
	det_file = os.path.join(output_dir, 'detections.pkl')
	cfg_yaml = yaml.dump(cfg)
	save_object(
		dict(
			all_boxes = all_boxes,
			all_segms = all_segms,
			all_keyps = all_keyps,
			cfg = cfg_yaml
		), det_file
	)
	logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
	
	return all_boxes, all_segms, all_keyps


def test_net(
		args,
		dataset_name,
		proposal_file,
		output_dir,
		ind_range = None,
		gpu_id = 0):
	"""Run inference on all images in a dataset or over an index range of images
	in a dataset using a single GPU.
	"""
	assert not cfg.MODEL.RPN_ONLY, \
		'Use rpn_generate to generate proposals from RPN-only models'
	
	roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
		dataset_name, proposal_file, ind_range
	)
	model = initialize_model_from_cfg(args, gpu_id = gpu_id)
	num_images = len(roidb)
	num_classes = cfg.MODEL.NUM_CLASSES
	all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
	timers = defaultdict(Timer)
	
	# cfg.TEST.PROPOSALS_OUT or cfg.TEST.ANCHOR_OUT
	dict_all = {}
	with open("/nfs/project/libo_i/Boosting/data/cache/coco_2017_val_100_gt_roidb.pkl", 'rb') as fp:
		cached_roidb = pickle.load(fp)
	assert len(roidb) == len(cached_roidb)
	
	for i, entry in enumerate(roidb):
		im_name = os.path.splitext(os.path.basename(entry['image']))[0]
		gt_i = cached_roidb[i]['boxes']
		if cfg.TEST.PRECOMPUTED_PROPOSALS:
			# The roidb may contain ground-truth rois (for example, if the roidb
			# comes from the training or val split). We only want to evaluate
			# detection on the *non*-ground-truth rois. We select only the rois
			# that have the gt_classes field set to 0, which means there's no
			# ground truth.
			box_proposals = entry['boxes'][entry['gt_classes'] == 0]
			if len(box_proposals) == 0:
				continue
		else:
			# Faster R-CNN type models generate proposals on-the-fly with an
			# in-network RPN; 1-stage models don't require proposals.
			box_proposals = None
		
		im = cv2.imread(entry['image'])
		cls_boxes_i, cls_segms_i, cls_keyps_i, im_scale = im_detect_all(model, im, box_proposals, timers)
		
		extend_results(i, all_boxes, cls_boxes_i)
		if cls_segms_i is not None:
			extend_results(i, all_segms, cls_segms_i)
		if cls_keyps_i is not None:
			extend_results(i, all_keyps, cls_keyps_i)
		
		if i % 10 == 0:  # Reduce log file size
			ave_total_time = np.sum([t.average_time for t in timers.values()])
			eta_seconds = ave_total_time * (num_images - i - 1)
			eta = str(datetime.timedelta(seconds = int(eta_seconds)))
			det_time = (
					timers['im_detect_bbox'].average_time +
					timers['im_detect_mask'].average_time +
					timers['im_detect_keypoints'].average_time
			)
			misc_time = (
					timers['misc_bbox'].average_time +
					timers['misc_mask'].average_time +
					timers['misc_keypoints'].average_time
			)
			logger.info(
				(
					'im_detect: range [{:d}, {:d}] of {:d}: '
					'{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
				).format(
					start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
					start_ind + num_images, det_time, misc_time, eta
				)
			)
		
		if cfg.TEST.PROPOSALS_OUT or cfg.TEST.ANCHOR_OUT:
			path = "/nfs/project/libo_i/Boosting/Anchor_Info"
			k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
			k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
			
			dict_i = {}
			
			if cfg.TEST.ANCHOR_OUT:
				if cfg.TEST.ANCHOR_VIS:
					with open(os.path.join(path, "anchor_5.json"), "r") as fp:
						anchor_lvl = json.load(fp)
					# Draw stage1 pred_boxes onto im and gt
					dpi = 200
					fig = plt.figure(frameon = False)
					fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
					ax = plt.Axes(fig, [0., 0., 1., 1.])
					ax.axis('off')
					fig.add_axes(ax)
					ax.imshow(im[:, :, ::-1])
					# 在im上添加gt
					for item in gt_i:
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'r',
							              linewidth = 0.6, alpha = 1))
					
					# 在im上添加proposals
					length = len(anchor_lvl)
					for ind in range(1050, 1060):
						item = anchor_lvl[ind]
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'g',
							              linewidth = 0.5, alpha = 0.8))
						ax.text(
							item[0], item[1] - 2,
							str(ind),
							fontsize = 3,
							family = 'serif',
							bbox = dict(
								facecolor = 'red', alpha = 1, pad = 0, edgecolor = 'none'),
							color = 'white')
					
					fig.savefig("/nfs/project/libo_i/Boosting/anchor_im_info/{}.png".format(im_name), dpi = dpi)
					plt.close('all')
				
				with open(os.path.join(path, "anchor_5.json"), "r") as fp:
					all_anchors = json.load(fp)
				
				all_anchors = np.array(all_anchors, dtype = np.float32)
				logger.info("Anchors num: {}".format(all_anchors.shape[0]))
				
				all_anchors = all_anchors / im_scale
				all_anchors.dtype = np.float32
				
				iou_mat = box_utils.bbox_overlaps(all_anchors, gt_i)
				max_inds = np.argmax(iou_mat, axis = 1)
				max_element = np.max(iou_mat, axis = 1)
				
				thrsh_inds = np.where(max_element > 0.5)
				max_inds = max_inds[thrsh_inds]
				all_anchors = all_anchors[thrsh_inds]
				
				center_point_distance = []
				iou_over_gt = []
				
				for ind, item in enumerate(all_anchors):
					# item是迭代中的pp, matched_gt是迭代中的ground_truth
					matched_gt_ind = max_inds[int(ind)]
					matched_gt = gt_i[matched_gt_ind]
					gt_center_point = ((matched_gt[2] - matched_gt[0] + 1) / 2, (matched_gt[3] - matched_gt[1] + 1)
					                   / 2)
					ac_center_point = ((item[2] - item[0]) / 2, (item[3] - item[1]) / 2)
					distance = np.sqrt(np.square(gt_center_point[0] - ac_center_point[0]) + np.square(
						gt_center_point[1] - ac_center_point[1]))
					
					distance = distance / ((item[2] - item[0]) / 2 + (matched_gt[2] - matched_gt[0]) / 2)
					# 计算intersect面积
					center_point_distance.append(distance)
					iw = min(item[2], matched_gt[2]) - max(item[0], matched_gt[0]) + 1
					intersect = 0
					if iw > 0:
						ih = min(item[3], matched_gt[3]) - max(item[1], matched_gt[1]) + 1
						if ih > 0:
							intersect = iw * ih
					
					gt_area = (matched_gt[2] - matched_gt[0] + 1) * (matched_gt[3] - matched_gt[1] + 1)
					assert gt_area > 0
					iou_over_gt.append(intersect / gt_area)
				
				center_point_distance = np.array(center_point_distance, dtype = np.float32)
				iou_over_gt = np.array(iou_over_gt, dtype = np.float32)
				dict_i['ac_center_point_distance'] = center_point_distance
				dict_i['ac_iou_over_gt'] = iou_over_gt
			
			if cfg.TEST.PROPOSALS_OUT:
				with open(os.path.join(path, "proposals.json"), "r") as fp:
					all_proposals = np.array(json.load(fp), dtype = np.float32)
				
				with open(os.path.join(path, "boxes.json"), "r") as fp:
					shifted_boxes = np.array(json.load(fp), dtype = np.float32)
				
				all_proposals = all_proposals / im_scale[0]
				all_proposals.dtype = np.float32
				
				shifted_boxes = shifted_boxes / im_scale[0]
				shifted_boxes.dtype = np.float32
				
				if cfg.TEST.ANCHOR_VIS:
					# Draw stage1 pred_boxes onto im and gt
					dpi = 200
					fig = plt.figure(frameon = False)
					fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
					ax = plt.Axes(fig, [0., 0., 1., 1.])
					ax.axis('off')
					fig.add_axes(ax)
					ax.imshow(im[:, :, ::-1])
					# 在im上添加gt
					for item in gt_i:
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'r',
							              linewidth = 0.6, alpha = 1))
					
					# 在im上添加proposals
					length = len(shifted_boxes)
					for ind in range(length):
						item = shifted_boxes[ind]
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'g',
							              linewidth = 0.3, alpha = 1))
					
					fig.savefig("/nfs/project/libo_i/Boosting/anchor_im_info/{}.png".format(im_name), dpi = dpi)
					plt.close('all')
				# center_point_lvl = []
				# for item in proposals_lvl:
				# 	center_point_lvl.append(((item[2] - item[0]) / 2, (item[3] - item[0]) / 2))
				
				if gt_i.shape[0] == 0:
					continue
				
				assert all_proposals.shape[0] == shifted_boxes.shape[0]
				pp_IOU, _, pp_IOG = stats_calculator(all_proposals, gt_i)
				dict_i['pp_IOU'] = pp_IOU.tolist()
				dict_i['pp_IOG'] = pp_IOG.tolist()
				
				bb_IOU, _, bb_IOG = stats_calculator(shifted_boxes, gt_i)
				dict_i['bb_IOU'] = bb_IOU.tolist()
				dict_i['bb_IOG'] = bb_IOG.tolist()
				
				assert pp_IOU.shape[0] == bb_IOU.shape[0]
				
				del _
			
			dict_all[im_name] = dict_i
		
		if i == 99:
			with open(os.path.join(path, "regression_info_norm_train200.json"), "w") as fp:
				json.dump(dict_all, fp)
		
		if cfg.VIS:
			im_name = os.path.splitext(os.path.basename(entry['image']))[0]
			vis_utils.vis_one_image(
				im[:, :, ::-1],
				'{:d}_{:s}'.format(i, im_name),
				os.path.join(output_dir, 'vis'),
				cls_boxes_i,
				segms = cls_segms_i,
				keypoints = cls_keyps_i,
				thresh = cfg.VIS_TH,
				box_alpha = 0.8,
				dataset = dataset,
				show_class = True
			)
	cfg_yaml = yaml.dump(cfg)
	if ind_range is not None:
		det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
	else:
		det_name = 'detections.pkl'
	det_file = os.path.join(output_dir, det_name)
	save_object(
		dict(
			all_boxes = all_boxes,
			all_segms = all_segms,
			all_keyps = all_keyps,
			cfg = cfg_yaml
		), det_file
	)
	logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
	return all_boxes, all_segms, all_keyps


def initialize_model_from_cfg(args, gpu_id = 0):
	"""Initialize a model from the global cfg. Loads test-time weights and
	set to evaluation mode.
	"""
	model = model_builder.Generalized_RCNN()
	model.eval()
	
	if args.cuda:
		model.cuda()
	
	if args.load_ckpt:
		load_name = args.load_ckpt
		logger.info("loading checkpoint %s", load_name)
		checkpoint = torch.load(load_name, map_location = lambda storage, loc:storage)
		net_utils.load_ckpt(model, checkpoint['model'])
	
	if args.load_detectron:
		logger.info("loading detectron weights %s", args.load_detectron)
		load_detectron_weight(model, args.load_detectron)
	
	model = mynn.DataParallel(model, cpu_keywords = ['im_info', 'roidb'], minibatch = True)
	
	return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
	"""Get the roidb for the dataset specified in the global cfg. Optionally
	restrict it to a range of indices if ind_range is a pair of integers.
	"""
	dataset = JsonDataset(dataset_name)
	if cfg.TEST.PRECOMPUTED_PROPOSALS:
		assert proposal_file, 'No proposal file given'
		roidb = dataset.get_roidb(
			proposal_file = proposal_file,
			proposal_limit = cfg.TEST.PROPOSAL_LIMIT
		)
	else:
		roidb = dataset.get_roidb()
	
	if ind_range is not None:
		total_num_images = len(roidb)
		start, end = ind_range
		roidb = roidb[start:end]
	else:
		start = 0
		end = len(roidb)
		total_num_images = end
	
	return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
	"""Return empty results lists for boxes, masks, and keypoints.
	Box detections are collected into:
	  all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
	Instance mask predictions are collected into:
	  all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
	  1:1 correspondence with the boxes in all_boxes[cls][image]
	Keypoint predictions are collected into:
	  all_keyps[cls][image] = [...] list of keypoints results, each encoded as
	  a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
	  [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
	  Keypoints are recorded for person (cls = 1); they are in 1:1
	  correspondence with the boxes in all_boxes[cls][image].
	"""
	# Note: do not be tempted to use [[] * N], which gives N references to the
	# *same* empty list.
	all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
	"""Add results for an image to the set of all results at the specified
	index.
	"""
	# Skip cls_idx 0 (__background__)
	for cls_idx in range(1, len(im_res)):
		all_res[cls_idx][index] = im_res[cls_idx]
