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


class Spco2ObjectFeature():
    def __init__(self):
        self.detect_object_info = []
        self.object_list = []
        self.dictionary = []
        self.cv_bridge = CvBridge()

    def object_server(self, trialname, step):
        if (os.path.exists(datafolder + trialname + "/tmp_boo/" + str(step - 1) + "_object_dic.csv") == True):
            with open(datafolder + trialname + "/tmp_boo/" + str(step - 1) + "_object_dic.csv", "r") as file:
                file_data = file.readlines()
                for line in file_data:
                    data = line.replace("\n", "")
                    self.dictionary.append(data)
            print("object_dic: {}".format(self.dictionary))

        self.taking_single_image(trialname, step)
        bb = rospy.wait_for_message('/yolov5_ros/output/bounding_boxes', BoundingBoxes, timeout=15)
        self.detect_object_info = bb.bounding_boxes
        if len(self.detect_object_info) == 0:  # histの部分だけあとで考える
            self.hist_w = np.zeros((1, 1))
            return self.object_list, self.dictionary, self.hist_w

        self.extracting_label()
        self.make_object_dic()
        self.hist_w = np.zeros((1, len(self.dictionary)))
        self.make_object_boo()
        return self.object_list, self.dictionary, self.hist_w

    def taking_single_image(self, trialname, step):
        img = rospy.wait_for_message('/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed', CompressedImage,
                                     timeout=None)
        observed_img = self.cv_bridge.compressed_imgmsg_to_cv2(img)
        cv2.imwrite(datafolder + trialname + "/object_image/" + str(step) + ".jpg", observed_img)
        return

    def extracting_label(self):
        for i in range(len(self.detect_object_info)):
            self.object_list.append(self.detect_object_info[i].Class)
        return

    def make_object_dic(self):
        for i in range(len(self.object_list)):
            if self.object_list[i] not in self.dictionary:
                self.dictionary.append(self.object_list[i])
        return

    def make_object_boo(self):
        for i in enumerate(self.object_list):
            idx = self.dictionary.index(i)
            self.hist_w[idx] += 1
        return


if __name__ == '__main__':
    rospy.init_node('spco2_object_feature')
    Spco2ObjectFeature()
    rospy.spin()
