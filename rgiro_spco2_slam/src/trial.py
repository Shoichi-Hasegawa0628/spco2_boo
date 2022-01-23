#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import shutil
import numpy as np

# with open("/root/HSR/catkin_ws/src/spco2_boo/rgiro_spco2_slam/data/output/" + "test" + "/tmp_boo/" + str(2) + "_Object_W_list.csv", 'r') as f:
#     reader = csv.reader(f)
#     Object_W_list = [row for row in reader]
#     print(Object_W_list)
#
# with open("/root/HSR/catkin_ws/src/spco2_boo/rgiro_spco2_slam/data/output/" + "test" + "/tmp_boo/" + str(2) + "_Object_BOO.csv", 'r') as f:
#     # reader = csv.reader(f)
#     reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
#     OT = [row for row in reader]
#     print(OT)
#
# # with open("/root/HSR/catkin_ws/src/spco2_boo/rgiro_spco2_slam/data/output/" + "test" + "/max_likelihood_param/1/" + "theta.csv", 'r') as f:
# #     # reader = csv.reader(f)
# #     reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
# #     theta = [row for row in reader]
# #     print(theta)
# #     print(sum(theta[0]))
#
# print("OT: {}".format(OT))
# print("Object_W_list: {}".format(Object_W_list))
# print(len(Object_W_list))
#
# # Cの場所概念にC^oのカウントを足す
# Nld_c = [sum([np.array(OT[0])])]
# print("Nld_c :{}".format(Nld_c))
# Xi = [(np.array(Nld_c[0]) + 1.0) / (sum(Nld_c[0]) + len(Object_W_list[0]) * 1.0)]
# print("Bunshi :{}".format((np.array(Nld_c[0]) + 1.0)))
# print("Bunbo :{}".format((sum(Nld_c[0]) + len(Object_W_list) * 1.0)))
# print("Xi :{}".format(Xi))

word = []
for i in range(80):
    FilePath = '/root/HSR/tmp/Otb_{}.csv'.format(i)
    with open(FilePath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(word)
    shutil.copyfile('/root/HSR/tmp/Otb_{}.csv'.format(i), '/root/HSR/tmp/Otb_{}.csv'.format(i+1))



