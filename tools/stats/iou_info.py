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

pp_DoC = []
pp_IOU = []
pp_IOG = []

for item in dict_all:
	pp_DoC_i = dict_all[item]["pp_DoC"]
	pp_IOU_i = dict_all[item]['pp_IOU']
	pp_IOG_i = dict_all[item]['pp_IOG']
	
	pp_IOG.extend(pp_IOG_i)
	pp_IOU.extend(pp_IOU_i)
	pp_DoC.extend(pp_DoC_i)

plt.subplot(211)
plt.title("Proposals Norm")
plt.scatter(pp_DoC, pp_IOU, c = "b", alpha = 0.5, s = 0.05)
plt.xlabel("Center Point Distance", fontsize = 8)
plt.ylabel("IOU", fontsize = 8)
plt.grid()

plt.subplot(212)
plt.scatter(pp_DoC, pp_IOG, c = "orange", alpha = 0.5, s = 0.05)
plt.xlabel("Center Point Distance", fontsize = 8)
plt.ylabel("IOG", fontsize = 8)
plt.grid()

plt.savefig("/nfs/project/libo_i/Boosting/{}.png".format(method), dpi = 200)
