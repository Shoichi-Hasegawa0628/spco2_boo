#! /usr/bin/env python
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
import subprocess
#from PIL import Image # PIL

from __init__ import *
from rgiro_spco2_slam.srv import spco_data_image,spco_data_imageResponse
import spco2_placescnn as places365

class ImageFeatureServer():

    def image_server(self, req):

        if len(self.frame) == 0:
             return spco_data_imageResponse(False)
        #cv2.imshow("image", self.frame)

        # forward pass
        convert_img = places365.Image.fromarray(self.frame)#convert into PIL
        input_img = places365.V(self.tf(convert_img).unsqueeze(0))
        logit = self.model.forward(input_img)
        h_x = places365.F.softmax(logit, 1).data.squeeze()
        
        # save image feature
        fp = open(self.DATA_FOLDER + '/img/ft' + str(req.count) + '.csv','a') 
        h_x_numpy = h_x.to('cpu').detach().numpy().copy()
        fp.write(','.join(map(str, h_x_numpy)))
        fp.write('\n')
        fp.close()
        rospy.loginfo("[Service] save new feature")

        # save image
        if self.image_save:
            if req.mode == "new":
                p = subprocess.Popen("mkdir " + self.DATA_FOLDER + "/image/", shell=True)
                rospy.sleep(0.5)
            image_name = self.DATA_FOLDER + "/image/" + str(req.count) + ".jpg"

            cv2.imwrite(image_name, self.frame)
            rospy.loginfo("[Service spco_data/image] save new image as %s", image_name)

        # save and publish activation image
        #print "h_x",h_x
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        # generate class activation mapping
        #print('Class activation map is saved as cam.jpg')
        #CAMs = places365.returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        #img = cv2.imread('test.jpg')
        '''
        height, width, _ = self.frame.shape#  img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img * 0.5
        image_name = self.DATA_FOLDER + "/image/" + str(req.count) + "_activation.jpg"
        cv2.imwrite(image_name, result)
        '''
        return spco_data_imageResponse(True)

    def image_callback(self, image):

        try:
            self.frame = CvBridge().imgmsg_to_cv2(image, "bgr8")
        except CvBrideError as e:
            print (e)

    def load_network_model(self):
        # load the labels
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = places365.load_labels()

        # load the model
        self.model = places365.load_model()

        # load the transformer
        self.tf = places365.returnTF() # image transformer

        # get the softmax weight
        self.params = list(self.model.parameters())
        self.weight_softmax = self.params[-2].data.numpy()
        self.weight_softmax[self.weight_softmax<0] = 0

        return (True)

    def __init__(self):

        TRIALNAME = "test"#rospy.get_param('~trial_name')#test
        IMAGE_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_raw" #"/camera/rgb/image_raw"#rospy.get_param('~image_topic')#/camera/rgb/image_raw
        self.image_save = True #rospy.get_param('~image_save')#true

        # subscrib image
        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        if self.load_network_model()==False:
            print ("error")

        self.DATA_FOLDER = datafolder + TRIALNAME
        self.frame = []
        
        s = rospy.Service('rgiro_spco2_slam/image', spco_data_image, self.image_server)
        rospy.loginfo("[Service spco_data/image] Ready")

if __name__ == "__main__":
    rospy.init_node('spco2_image_features',anonymous=False)
    srv = ImageFeatureServer()
    rospy.spin()
