import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from FFAG_Bunch import FFAG_ManageBunchAttribute
import argparse
from FFAG_ParasAndConversion import FFAG_ConversionTools

data = pd.read_csv("./sc10/step_dump_sc_1/merged_files/particle_4_rank_merged.csv", header=None).to_numpy()
plt.figure()
plt.plot(data[:, 0] * np.cos(data[:, 4]), data[:, 0] * np.sin(data[:, 4]))

plt.figure()
plt.plot(data[:, 2])
plt.show()
# foldpath = ".\\step_dump_sc_3\\merged_files"
# attributes = FFAG_ManageBunchAttribute().Attribute  # 获取属性字典
# # 获取所有穿越点文件（假设以 .csv 保存）
# crossing_files = [f for f in os.listdir(foldpath) if f.endswith('.csv')]
#
# for i, crossing_file in enumerate(crossing_files):
#     filepath = os.path.join(foldpath, crossing_file)
#
#     # 加载穿越点数据
#     data = pd.read_csv(filepath, header=None).to_numpy()
#     data_p = FFAG_ConversionTools().ConvertVrzek2Przek(data[:, :15])
#
#     # 按属性提取数据
#     if np.size(data_p, 0) > 9:
#         fi = data_p[2, attributes['fi']]  # 径向位置
#         r = data_p[2, attributes['r']]  # 径向位置
#         z = data_p[2, attributes['z']]  # 垂直位置
#         pr = data_p[2, attributes['vr']]  # 径向速度
#         pz = data_p[2, attributes['vz']]  # 垂直速度
#         plt.figure(1)
#         plt.scatter(r * 1000, pr * 1000, s=3.0, alpha=0.8, color='blue')
#         plt.figure(2)
#         plt.scatter(z * 1000, pz * 1000, s=3.0, alpha=0.8, color='blue')
#         plt.figure(3)
#         plt.scatter(fi * 1000, z * 1000, s=3.0, alpha=0.8, color='blue')
#
# plt.xlabel("r(mm)")
# plt.ylabel("pr(mrad)")
# plt.show()
