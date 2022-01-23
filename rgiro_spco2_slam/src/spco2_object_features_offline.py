#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Library
from __future__ import unicode_literals
import codecs
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import csv

# Third Party
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from yolo_ros_msgs.msg import BoundingBoxes, BoundingBox

# Self Modules
from __init__ import *
from rgiro_spco2_slam.srv import spco_data_object, spco_data_objectResponse


class ObjectFeatureServer():
    def __init__(self):
        self.DATA_FOLDER = "/root/HSR/catkin_ws/src/spco2_boo/rgiro_spco2_slam/data"
        self.detect_object_info = []
        self.object_list = []
        self.Object_BOO = []
        self.cv_bridge = CvBridge()
        # o = rospy.Service('rgiro_spco2_slam/object', spco_data_object, self.object_server)
        rospy.loginfo("[Service spco_data/object] Ready")
        self.object_server()

    def object_server(self):
        # trialname = "test"
        files = os.listdir(self.DATA_FOLDER + "/raw")
        for i in range(len(files)):

            if (os.path.exists(self.DATA_FOLDER + "/tmp_boo/Object.csv") == True):
                with open(self.DATA_FOLDER + "/tmp_boo/Object.csv", 'r') as f:
                    reader = csv.reader(f)
                    self.object_list = [row for row in reader]
                print("pre_object_list: {}\n".format(self.object_list))

            bb = rospy.wait_for_message('/yolov5_ros/output/bounding_boxes', BoundingBoxes, timeout=15)
            self.detect_object_info = bb.bounding_boxes
            # print(self.detect_object_info)

            if len(self.detect_object_info) == 0:
                if i + 1 == 1:
                    # 最初の教示で物体が検出されなかったとき
                    self.object_list = [[]]
                    self.Object_BOO = [[0] * len(object_dictionary)]
                    # self.taking_single_image(trialname, req.step)
                    self.save_data(i + 1)
                    # return spco_data_objectResponse(True)

                else:
                    # 最初の教示以降の教示で物体が検出されなかったとき
                    object_list = []
                    self.object_list.append(object_list)
                    self.make_object_boo()
                    # self.taking_single_image(trialname, req.step)
                    self.save_data(i + 1)
                    # return spco_data_objectResponse(True)
            else:
                self.save_detection_img(i + 1)
                self.extracting_label()
                self.make_object_boo()
                # self.taking_single_image(trialname, req.step)
                self.save_data(i + 1)
                print("object_list: {}\n".format(self.object_list))
                print("dictionary: {}\n".format(object_dictionary))
                print("Bag-of-Objects: {}\n".format(self.Object_BOO))
                # return spco_data_objectResponse(True)

    def extracting_label(self):
        object_list = []
        for i in range(len(self.detect_object_info)):
            object_list.append(self.detect_object_info[i].Class)
            print(object_list)
        self.object_list.append(object_list)
        print(self.object_list)
        return

    def make_object_boo(self):
        # print(self.object_list)
        self.Object_BOO = [[0 for i in range(len(object_dictionary))] for n in range(len(self.object_list))]
        # print(self.Object_BOO)
        for n in range(len(self.object_list)):
            for j in range(len(self.object_list[n])):
                for i in range(len(object_dictionary)):
                    if object_dictionary[i] == self.object_list[n][j]:
                        self.Object_BOO[n][i] = self.Object_BOO[n][i] + 1
        # print(self.Object_BOO)
        return

    # def taking_single_image(self, trialname, step):
    #     img = rospy.wait_for_message('/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed', CompressedImage,
    #                                  timeout=None)
    #     observed_img = self.cv_bridge.compressed_imgmsg_to_cv2(img)
    #     cv2.imwrite(datafolder + trialname + "/object_image/" + str(step) + ".jpg", observed_img)
    #     return

    def save_detection_img(self, step):
        img = rospy.wait_for_message('/yolov5_ros/output/image/compressed', CompressedImage, timeout=15)
        detect_img = self.cv_bridge.compressed_imgmsg_to_cv2(img)
        cv2.imwrite(self.DATA_FOLDER + "/detect_image/" + str(step) + ".jpg", detect_img)
        return

    def save_data(self, step):
        # 全時刻の観測された物体のリストを保存
        FilePath = self.DATA_FOLDER + "/tmp_boo/Object.csv"
        with open(FilePath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.object_list)

        # 教示ごとに観測された物体のリストを保存
        FilePath = self.DATA_FOLDER + "/tmp_boo/" + str(step) + "_Object.csv"
        with open(FilePath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.object_list)

        # 教示ごとのBag-Of-Objects特徴量を保存
        FilePath = self.DATA_FOLDER + "/tmp_boo/" + str(step) + "_Object_BOO.csv"
        with open(FilePath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.Object_BOO)

        # 教示ごとの物体の辞書を保存
        FilePath = self.DATA_FOLDER + "/tmp_boo/" + str(step) + "_Object_W_list.csv"
        with open(FilePath, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(object_dictionary)

        return


if __name__ == '__main__':
    rospy.init_node('spco2_object_features', anonymous=False)
    srv = ObjectFeatureServer()
    # rospy.spin()
