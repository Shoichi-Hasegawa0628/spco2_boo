#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

# Simple listener that listens to std_msgs/Strings 'speech_to_text' topic and output csv

import rospy
from std_msgs.msg import String
import std_msgs.msg
import std_srvs.srv
import csv
import time

from rgiro_spco2_slam.srv import spco_data_image


def callback(message):
    rospy.loginfo(rospy.get_caller_id() + ' I heard %s', message.data)
    OutputString = message.data.split()

    # save massage as a csv format
    FilePath = '/root/HSR/catkin_ws/src/spco_library/rgiro_spco2_slam/data/output/test/tmp/Otb.csv'
    with open(FilePath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(OutputString)

    # request image feature
    service_result = False
    step = sum([1 for _ in open(FilePath)])
    print ("step=",step)

    rospy.wait_for_service('rgiro_spco2_slam/image')
    srv = rospy.ServiceProxy('rgiro_spco2_slam/image', spco_data_image)
    try:
        service_result = srv("new", step)
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))

    ## Publish messeage for start SpCo leaning.
    pub = rospy.Publisher('start_learning', std_msgs.msg.String, queue_size=10, latch=True)
    str_msg = std_msgs.msg.String(data= message.data )
    rospy.loginfo('%s publish %s'%(rospy.get_name(),str_msg.data))
    pub.publish(str_msg)

    
def listener():

    ## node initalization
    rospy.init_node('listener_node', anonymous=False)

    ## subscriber
    rospy.Subscriber('speech_to_text', String, callback)
    ## service
    #rospy.Service('speech_to_text', std_srvs.srv.Trigger, service_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
