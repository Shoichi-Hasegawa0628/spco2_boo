#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Library
from __future__ import unicode_literals
import codecs
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Third Party
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from yolo_ros_msgs.msg import BoundingBoxes, BoundingBox


class Spco2ObjectFeature():
    def __init__(self):
        self.detect_object_info = []
        self.object_list = []
        self.dictionary = []
        self.cv_bridge = CvBridge()

    def object_server(self):
        ## もし辞書があればここでロードする
        img = self.taking_single_image()
        bb = rospy.wait_for_message('/yolov5_ros/output/bounding_boxes', BoundingBoxes, timeout=15)
        self.detect_object_info = bb.bounding_boxes
        self.extracting_label()
        self.make_object_dic()
        self.hist_w = np.zeros((1, len(self.dictionary)))
        self.make_object_boo()
        return

    ## 画像を保存するパスの設定
    def taking_single_image(self):
        img = rospy.wait_for_message('/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed', CompressedImage,
                                     timeout=None)
        observed_img = self.cv_bridge.compressed_imgmsg_to_cv2(img)
        cv2.imwrite('observed_img.jpg', observed_img)
        return img

    def extracting_label(self):
        for i in range(len(self.detect_object_info)):
            self.object_list.append(self.detect_object_info[i].Class)
        return

    ## 辞書を保存するパスの設定
    def make_object_dic(self):
        for i in range(len(self.object_list)):
            if self.object_list[i] not in self.dictionary:
                self.dictionary.append(self.object_list[i])

        codecs.open("word_dic.txt", "w", "utf-8").write("\n".join(self.dictionary))
        rospy.loginfo("Saved the word dictionary as %s\n", "word_dic.txt")
        return

    ## booを保存するパスの設定
    def make_object_boo(self):
        for i in enumerate(self.object_list):
            # print(word)
            idx = self.dictionary.index(i)
            self.hist_w[idx] += 1

        np.savetxt("histgram_object.txt", self.hist_w, fmt=str("%d"))
        rospy.loginfo("Saved the word histgram as %s\n", "histgram_object.txt")
        return


if __name__ == '__main__':
    rospy.init_node('spco2_object_feature')
    Spco2ObjectFeature()
