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
	
plt.subplot(211)
plt.title("Val 200/Purple:IOG,Red:IOU")
plt.scatter(pp_IOU, bb_IOG, c = "blue", alpha = 0.5, s = 0.05)
plt.ylabel("Boxes IOG", fontsize = 8)
plt.grid()

plt.subplot(212)
plt.tick_params(axis = 'both', which = 'minor', labelsize = 4)
plt.plot(x_line, IOU_mean_value, c = 'r')
plt.plot(x_line, IOG_mean_value, c = 'purple')
plt.scatter(pp_IOU, bb_IOU, c = "orange", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOU", fontsize = 8)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.grid()

plt.savefig("/nfs/project/libo_i/Boosting/{}_1.png".format(method), dpi = 200)
plt.close('all')

plt.subplot(211)
plt.title("Val 200/Purple:IOG,Red:IOU")
plt.scatter(pp_IOG, bb_IOU, c = "blue", alpha = 0.5, s = 0.05)
plt.ylabel("Boxes IOU", fontsize = 8)
plt.grid()

plt.subplot(212)
plt.plot(x_line, IOU_mean_value, c = 'r')
plt.plot(x_line, IOG_mean_value, c = 'purple')
plt.scatter(pp_IOG, bb_IOG, c = "orange", alpha = 0.5, s = 0.05)
plt.xlabel("Proposals IOG", fontsize = 8)
plt.ylabel("Boxes IOG", fontsize = 8)
plt.grid()

plt.savefig("/nfs/project/libo_i/Boosting/{}_2.png".format(method), dpi = 200)
plt.close('all')

# plt.title("Val 200/Purple:IOG,Red:IOU")
#
# # plt.scatter(pp_IOU, bb_IOU, c = "blue", alpha = 0.5, s = 0.05)
# # plt.xlabel("Before IOU", fontsize = 8)
# # plt.ylabel("After IOU", fontsize = 8)
# # plt.grid()
#
# plt.plot(x_line, IOG_mean_value, c = 'purple')
# # plt.scatter(pp_IOG, bb_IOG, c = "orange", alpha = 0.5, s = 0.05)
# plt.xlabel("Before", fontsize = 8)
# plt.ylabel("After", fontsize = 8)
# plt.grid()
#
# plt.savefig("/nfs/project/libo_i/Boosting/{}_BA.png".format(method), dpi = 200)
# plt.close('all')
print("OK")
