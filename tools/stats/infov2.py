from scipy import stats
import numpy as np
import os
import json
import sys

sys.path.insert(0, '/nfs/project/libo_i/Boosting/lib')
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

method = "regression_info_norm"

file_path = "/nfs/project/libo_i/Boosting/Anchor_Info/{}.json".format(method)
with open(file_path, 'r') as fp:
	dict_all = json.load(fp)

pp_IOU = []
pp_IOG = []
bb_IOU = []
bb_IOG = []

for item in dict_all:
	pp_IOU_i = dict_all[item]['pp_IOU']
	pp_IOG_i = dict_all[item]['pp_IOG']
	
	bb_IOU_i = dict_all[item]['bb_IOU']
	bb_IOG_i = dict_all[item]['bb_IOG']
	
	pp_IOG.extend(pp_IOG_i)
	pp_IOU.extend(pp_IOU_i)
	
	bb_IOU.extend(bb_IOU_i)
	bb_IOG.extend(bb_IOG_i)

assert len(pp_IOU) == len(bb_IOU)
assert len(pp_IOG) == len(bb_IOG)

sorted_pp_IOU = np.sort(pp_IOU)
sorted_index = np.argsort(pp_IOU)
sorted_bb_IOU = np.array(bb_IOU, dtype = np.float32)[sorted_index]

sorted_pp_IOG = np.sort(pp_IOG)
sorted_index_IOG = np.argsort(pp_IOG)
sorted_bb_IOG = np.array(bb_IOG, dtype = np.float32)[sorted_index_IOG]

IOU_mean_value = []
x_line = np.linspace(0, 1, 101)[0:-1]
for item in x_line:
	indx_left = np.searchsorted(sorted_pp_IOU, item)
	indx_right = np.searchsorted(sorted_pp_IOU, item + 0.1)
	if indx_right >= len(sorted_bb_IOU):
		indx_right = len(sorted_bb_IOU) - 1
	
	IOU_mean_value.append(
		np.mean(sorted_bb_IOU[indx_left: indx_right + 1]))
x_line += 0.05

IOG_mean_value = []
x_line = np.linspace(0, 1, 101)[0:-1]
for item in x_line:
	indx_left = np.searchsorted(sorted_pp_IOG, item)
	indx_right = np.searchsorted(sorted_pp_IOG, item + 0.1)
	if indx_right >= len(sorted_bb_IOG):
		indx_right = len(sorted_bb_IOG) - 1
	IOG_mean_value.append(
		np.mean(sorted_bb_IOG[indx_left: indx_right + 1]))

dataset_name = "Val 200"
plt.scatter(pp_IOU, pp_IOG, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOU", fontsize = 8)
plt.ylabel("Proposals IOG", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_1.png".format(method), dpi = 200)
plt.close('all')

plt.scatter(bb_IOU, bb_IOG, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Boxes IOU", fontsize = 8)
plt.ylabel("Boxes IOG", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_2.png".format(method), dpi = 200)
plt.close('all')

plt.scatter(pp_IOU, bb_IOU, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOU", fontsize = 8)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_3.png".format(method), dpi = 200)
plt.close('all')

plt.scatter(pp_IOU, bb_IOG, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOU", fontsize = 8)
plt.ylabel("Boxes IOG", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_4.png".format(method), dpi = 200)
plt.close('all')

plt.scatter(pp_IOG, bb_IOU, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOG", fontsize = 8)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_5.png".format(method), dpi = 200)
plt.close('all')

plt.scatter(pp_IOG, bb_IOG, c = "blue", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOG", fontsize = 8)
plt.ylabel("Boxes IOG", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_6.png".format(method), dpi = 200)
plt.close('all')

# 画给定阈值的曲线图了
# 找出 proposals IOU 在0.5-0.6区间的点，统计这些框的PIOG x BIOU的关系

pp_IOU = np.array(pp_IOU, dtype = np.float32)
pp_IOG = np.array(pp_IOG, dtype = np.float32)
bb_IOU = np.array(bb_IOU, dtype = np.float32)
bb_IOG = np.array(bb_IOG, dtype = np.float32)

piou_inds = np.intersect1d(np.where(pp_IOU <= 0.8)[0], np.where(pp_IOU >= 0.6)[0])

constraint_pIOG = pp_IOG[piou_inds]
constraint_bIOU = bb_IOU[piou_inds]
sorted_inds = np.argsort(constraint_pIOG)
sorted_cons_pIOG = constraint_pIOG[sorted_inds]
sorted_cons_bIOU = constraint_bIOU[sorted_inds]

constraint_IOU_mean_value = []
x_line = np.linspace(0.5, 1, 51)[0:-1]
for item in x_line:
	indx_left = np.searchsorted(sorted_cons_pIOG, item)
	indx_right = np.searchsorted(sorted_cons_pIOG, item + 0.01)
	if indx_right >= len(sorted_cons_pIOG):
		indx_right = len(sorted_cons_pIOG) - 1
	constraint_IOU_mean_value.append(np.mean(sorted_cons_bIOU[indx_left: indx_right + 1]))

x_line += 0.005

plt.scatter(constraint_pIOG, constraint_bIOU, c = "blue", alpha = 0.5, s = 0.05)
plt.plot(x_line, constraint_IOU_mean_value, c = 'purple', alpha = 0.5)
plt.xlabel("Proposals IOG", fontsize = 8)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.title("{}_.5_1".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_7.png".format(method), dpi = 200)
plt.close('all')

# 找出 proposals IOG 在0.5-0.6区间的点，统计这些框的PIOU x BIOU的关系
piog_inds = np.intersect1d(np.where(pp_IOG <= 0.6)[0], np.where(pp_IOG >= 0.5)[0])

constraint_pIOU = pp_IOG[piog_inds]
constraint_bIOU = bb_IOU[piog_inds]
sorted_inds = np.argsort(constraint_pIOU)
sorted_cons_pIOU = constraint_pIOU[sorted_inds]
sorted_cons_bIOU = constraint_bIOU[sorted_inds]

constraint_IOU_mean_value = []
x_line = np.linspace(0.5, 0.6, 11)[0:-1]
for item in x_line:
	indx_left = np.searchsorted(sorted_cons_pIOU, item)
	indx_right = np.searchsorted(sorted_cons_pIOU, item + 0.05)
	if indx_right >= len(sorted_cons_pIOU):
		indx_right = len(sorted_cons_pIOU) - 1
	constraint_IOU_mean_value.append(np.mean(sorted_cons_bIOU[indx_left: indx_right + 1]))

x_line += 0.005

plt.scatter(constraint_pIOU, constraint_bIOU, c = "blue", alpha = 0.5, s = 0.05)
plt.plot(x_line, constraint_IOU_mean_value, c = 'purple', alpha = 0.5)
plt.xlabel("Proposals IOU", fontsize = 8)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.title("{}".format(dataset_name))
plt.grid()
plt.savefig("/nfs/project/libo_i/Boosting/{}_8.png".format(method), dpi = 200)
plt.close('all')

print("OK")
