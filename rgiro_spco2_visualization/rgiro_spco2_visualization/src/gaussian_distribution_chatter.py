#! /usr/bin/env python

import glob
import math
import re
import csv
import rospy
import numpy as np
import time

from rgiro_spco2_visualization_msgs.msg import GaussianDistributions, GaussianDistribution
from rgiro_spco2_visualization_msgs.srv import GaussianService, GaussianServiceRequest
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import std_msgs.msg
import os

from __init__ import *


class EmSpcotRviz(object):

    def __init__(self):

        #self._frame_id = "/map"
        #self._before_cylinder_rviz_id_count = 0
        #self._before_text_rviz_id_count = 0
        #self._visualize_concepts = set()
        #self._word_list = np.loadtxt("parameter/name_dic.csv", delimiter="\n", dtype="S")  # dictionary of name

        #self._marker_pub = rospy.Publisher("em/draw_position/array", MarkerArray, queue_size=1)
        #rospy.Subscriber("em_spco_transfer/visualize_concept", String, self.call_back, queue_size=10)
        #rospy.Service("em_spco_transfer/rviz", GaussianService , self.rviz_server_call_back)
        #rospy.Service("em_spco_transfer/rviz", spcot_rviz, self.rviz_server_call_back)
        #rospy.Publisher("em_spco_transfer/visualize_concept", String, self.call_back, queue_size=10)
        #pub = rospy.Publisher("gaussian_in",GaussianDistributions, queue_size=10)
        pub = rospy.Publisher("transfer_learning/gaussian_distribution",GaussianDistributions, queue_size=10)
        rospy.sleep(0.5)
        rospy.loginfo("start visualization!!")
        #time.sleep(1)

   #def call_back(self, msg):
        # type: (String) -> None
        #if msg.data in self._word_list:
            #self._visualize_concepts.add(msg.data)

#    def rviz_server_call_back(self, req):
    #def call_back():
        # type: (spcot_rvizRequest) -> spcot_rvizResponse

        # print self._visualize_concepts

        #if not req.start:
        #    return spcot_rvizResponse(False)

        ################################################################################################################
        #       Init Path, Load File
        ################################################################################################################
        step=0
        trialname="test"
        while os.path.exists(datafolder + trialname + '/' + str(step+1)):
            step += 1
        print ("step",step)

        if (LMweight != "WS"):
            omomi = '/weights.csv'
        else: #if (LMweight == "WS"):
            omomi = '/WS.csv'
        
        i = 0
        for line in open(datafolder + trialname + '/'+ str(step) + omomi, 'r'):
            #itemList = line[:-1].split(',')
            if (i == 0):
                max_particle = int(line)
                i += 1
 
        #filename = datafolder + trialname + '/' + str(step) + '/mu0.csv'
        filename = datafolder + trialname + '/' + str(step) + '/mu'+ str(max_particle) +'.csv'
        mu = []
        print("max_particle:",max_particle)
        print ("filename:",filename)
        mu = np.genfromtxt(filename,delimiter=',')
        #mu = np.genfromtxt("/root/RULO/catkin_ws/src/rgiro_spco2_slam/data/teachingtext/mu29.csv",delimiter=',')
        #mu = np.loadtxt("/root/RULO/catkin_ws/src/rgiro_spco2_slam/data/teachingtext/mu29.csv",delimiter=',')
        print ("mu",mu)
        #word = np.loadtxt(result_path + "/phi_n.csv")
        #pi = np.loadtxt(result_path + "/pi.csv")
        #Ct = np.loadtxt(result_path + "/Ct.csv")
        #rt = np.loadtxt(result_path + "/rt.csv")
        #pi_max = np.argmax(pi, axis=0)

        sigma = []
        filename = datafolder + trialname + '/' + str(step) + '/sig'+ str(max_particle) +'.csv'
        #filename = datafolder + trialname + '/' + str(step) + '/sig0.csv'
        sigma = np.genfromtxt(filename,delimiter=',' )
        #sigma = np.genfromtxt("/root/RULO/catkin_ws/src/rgiro_spco2_slam/data/teachingtext/sig29.csv",delimiter=',' )
        #print temp
        #files = glob.glob(result_path + "/sigma/*.csv")
        #convert = lambda text: int(text) if text.isdigit() else text
        #alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        #files.sort(key=alphanum_key)
        #for line in file:
        #    array = np.loadtxt(f)
        #    sigma.append(array)
        print ("sigma",sigma)

        #color = np.loadtxt("./color.csv", delimiter=",")
        script_dir_abspath = os.path.dirname(os.path.abspath(__file__))
        color = np.loadtxt(script_dir_abspath + "/color.csv", delimiter=",")
        #color = np.loadtxt(script_dir_abspath + "/root/RULO/catkin_ws/src/rgiro_spco2/rgiro_rgiro_spco2_visualization/rgiro_spco2_visualization/src/color.csv", delimiter=",")
        #print "color",color
        #word_index_dic = {}
        if mu.ndim == 1:
            region_num=1
        else:
            region_num = len(sigma)
        print ("num",region_num)
        #for i in range(len(self._word_list)):
        #    word_index_dic[self._word_list[i]] = i
        """
        ################################################################################################################
        #       Generate Markers
        ################################################################################################################
        marker_array = MarkerArray()
        cylinder_marker_id = 0
        region_num = np.unique(rt).astype(np.int)  # 20191118kamei Deduplicate with unique

        for i in region_num:

            if not (Ct[i] > 0):
                continue

            # w = self._word_list[np.argsort(word[pi_max[i]])[::-1][0]]
            # if w not in self._visualize_concepts:
            #     continue

            (eigValues, eigVectors) = np.linalg.eig(sigma[i])
            angle = (math.atan2(eigVectors[1, 0], eigVectors[0, 0]))

            cylinder_marker = Marker()
            cylinder_marker.type = Marker.CYLINDER
            cylinder_marker.ns = "cylinder"
            cylinder_marker.scale.x = math.sqrt(eigValues[0])  # 10*math.sqrt(eigValues[0])
            cylinder_marker.scale.y = math.sqrt(eigValues[1])  # 10*math.sqrt(eigValues[1])
            cylinder_marker.scale.z = 0.6 * word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
            cylinder_marker.pose.position.x = mu[i][0]
            cylinder_marker.pose.position.y = mu[i][1]
            cylinder_marker.pose.position.z = 0.3 * word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
            cylinder_marker.pose.orientation.w = math.cos(angle * 0.5)
            cylinder_marker.pose.orientation.z = math.sin(angle * 0.5)
            cylinder_marker.header.frame_id = self._frame_id
            cylinder_marker.header.stamp = rospy.Time.now()
            cylinder_marker.id = cylinder_marker_id
            cylinder_marker.action = Marker.ADD
            cylinder_marker.color.r = color[i][0] / 255.0
            cylinder_marker.color.g = color[i][1] / 255.0
            cylinder_marker.color.b = color[i][2] / 255.0
            cylinder_marker.color.a = 0
            marker_array.markers.append(cylinder_marker)
            cylinder_marker_id += 1

        height = 0
        text_marker_id = 0

        for i in region_num:

            if not (Ct[i] > 0):
                continue

            w = self._word_list[np.argsort(word[pi_max[i]])[::-1][0]]
            # if w not in self._visualize_concepts:
            #     continue

            loop = 3
            if len(self._word_list) - 1 < 3:
                loop = len(self._word_list) - 1

            for k in xrange(loop):
                output_text = "%s: %.f%%" % (self._word_list[np.argsort(word[pi_max[i]])[::-1][k]], word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][k]] * 100)
                word_index = word_index_dic[w]
                r = color[word_index][0] / 255.0
                g = color[word_index][1] / 255.0
                b = color[word_index][2] / 255.0

                text_marker = Marker()
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.header.frame_id = self._frame_id
                text_marker.ns = "text"
                text_marker.header.stamp = rospy.Time.now()
                text_marker.id = text_marker_id
                text_marker.action = Marker.ADD
                text_marker.scale.x = 0.12 - 0.02 * k
                text_marker.scale.y = 0.12 - 0.02 * k
                text_marker.scale.z = 0.12 - 0.02 * k
                text_marker.pose.position.x = mu[i][0]
                text_marker.pose.position.y = mu[i][1]
                text_marker.pose.position.z = 1.0 * word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]] + 0.5 - 0.3 * (0.4 * k)
                text_marker.color.r = min(1.0, r + 0.25)
                text_marker.color.g = min(1.0, g + 0.25)
                text_marker.color.b = min(1.0, b + 0.25)
                text_marker.color.a = 0.8
                text_marker.text = output_text
                marker_array.markers.append(text_marker)
                text_marker_id += 1
                height += 1

        if self._before_text_rviz_id_count > text_marker_id:
            for i in range(text_marker_id, self._before_text_rviz_id_count):
                delete_marker = Marker()
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.id = i
                delete_marker.ns = "text"
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)

        if self._before_cylinder_rviz_id_count > cylinder_marker_id:
            for i in range(cylinder_marker_id, self._before_cylinder_rviz_id_count):
                delete_marker = Marker()
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.id = i
                delete_marker.ns = "cylinder"
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
        """
        ################################################################################################################
        #       Publish Gaussian Distribution
        ################################################################################################################
        distributions = GaussianDistributions()
        
        # print()
        print("R = ", region_num)
        for i in range(region_num):
                #print "mu",mu
                #print "sigma",sigma
            #if Ct[i] > 0:
                #w = self._word_list[np.argsort(word[pi_max[i]])[::-1][0]]
                # if w not in self._visualize_concepts:
                #     continue

                word_index = i#word_index_dic[w]
                print ("i;",i)
                distribution = GaussianDistribution()
                if mu.ndim == 1:
                    distribution.mean_x = mu[0]
                    distribution.mean_y = mu[1]
                else:
                    distribution.mean_x = mu[i][0]
                    distribution.mean_y = mu[i][1]

                if sigma.ndim == 1:
                    distribution.variance_x = np.sqrt(sigma[0])
                    distribution.variance_y = np.sqrt(sigma[3])
                    distribution.covariance = sigma[1]#[0]
                    correlation_coefficient = (sigma[1])**2 / (sigma[0] * sigma[3])
                    print(np.pi)
                    print(correlation_coefficient)
                    print(sigma[0])
                    print(sigma[1])
                    print(sigma[2])
                    print(sigma[3])
                    distribution.probability = 1.0 / (2.0 * np.pi * np.sqrt(sigma[0] * sigma[3] * (1.0 - correlation_coefficient))) #word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
                else:
                    distribution.variance_x = np.sqrt(sigma[i][0])
                    distribution.variance_y = np.sqrt(sigma[i][3])
                    distribution.covariance = sigma[i][1]#[0]
                    correlation_coefficient = (sigma[i][1])**2 / (sigma[i][0] * sigma[i][3])
                    distribution.probability = 1.0 / (2.0 * np.pi * np.sqrt(sigma[i][0] * sigma[i][3] * (1.0 - correlation_coefficient))) #word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
                print(distribution.probability)
                distribution.r = int(color[word_index][0])
                distribution.g = int(color[word_index][1])
                distribution.b = int(color[word_index][2])
                distributions.distributions.append(distribution)
                # print(word_inedx, distribution.probability)

        # self.gaussian_pub.publish(distributions)
        rate = rospy.Rate(1) # 1 Hz
        #rospy.loginfo('%s publish %s'%(rospy.get_name(),distributions))
        pub.publish(distributions)
        rate.sleep()
 
        #rospy.wait_for_service("/em/spcot/gaussian_request")
        #service = rospy.ServiceProxy("/em/spcot/gaussian_request", GaussianService)
        #request = GaussianServiceRequest(distributions)
        #service(request)

        #self._marker_pub.publish(marker_array)

        #self._before_text_rviz_id_count = text_marker_id
        #self._before_cylinder_rviz_id_count = cylinder_marker_id

        #return spcot_rvizResponse(True)
def callback(message):
    #print "start visualization!"
    #while not rospy.is_shutdown():
    EmSpcotRviz()


if __name__ == "__main__":
    rospy.init_node("spcot_rviz", anonymous=False)
    print ("start visualization")
    #rate = rospy.Rate(1) # 1 Hz
    rospy.Subscriber('start_visualization', String, callback)
    #rospy.Subscriber('speech_to_text', String, callback)

    rospy.spin()
