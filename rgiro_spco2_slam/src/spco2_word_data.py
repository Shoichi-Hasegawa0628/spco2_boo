#!/usr/bin/env python
import rospy
import std_msgs.msg
#import std_srvs.srv
import readline

'''
def CallService():        
    ## service call
    service_result = None

    name = raw_input('Enter teaching text: ')
    print('name is ' + name)
    rospy.loginfo('waiting service %s'%(rospy.resolve_name('servicename')))
    rospy.wait_for_service('servicename')
    try:
        srv_prox = rospy.ServiceProxy('servicename',std_srvs.srv.Trigger)
        res = srv_prox()
        if res.success:
            service_result = res.message
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    name = raw_input('Enter teaching text: ')
    print('name is ' + name)
'''
def StartPublish(): 
    ## node initialization
    rospy.init_node('spco2_word_data_node', anonymous=False)
    pub = rospy.Publisher('speech_to_text', std_msgs.msg.String, queue_size=10) ## queue size is not important for sending just one messeage.
    rate = rospy.Rate(1) # 1 Hz
    while not rospy.is_shutdown():
        TeachingText = input('Enter teaching text: ')
        print('teaching text: ' + TeachingText)
        str_msg = std_msgs.msg.String(data= TeachingText )
        rospy.loginfo('%s publish %s'%(rospy.get_name(),str_msg.data))
        pub.publish(str_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        ## This function might be made as a sevice of ROS for stable work.
        ## Use publisher function tentatively,because it is easy.
        #CallService()
        StartPublish()
    except rospy.ROSInterruptException: pass