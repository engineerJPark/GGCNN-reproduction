#! /usr/bin/env python

import time
import numpy as np
# import argparse

import torch
import rospy
import rospkg

from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String, Bool, Int8

# def parse_args():
#     parser = argparse.ArgumentParser(description='GG-CNN Node')
#     parser.add_argument('--plot', action='store_true', help='Plot depth image')
# 	parser.add_argument('--gazebo', action='store_true', help='Set properties based on the Gazebo environment')
# 	args = parser.parse_args()
# 	return args
# args = parse_args()

######## time class

class TimeIt:
	def __init__(self, s):
		self.s = s
		self.t0 = None
		self.t1 = None
		self.print_output = False

	def __enter__(self):
		self.t0 = time.time()

	def __exit__(self, t, value, traceback):
		self.t1 = time.time()
		print('%s: %s' % (self.s, self.t1 - self.t0))

######## GGCNN

class ros_gg_cnn():
    def __init__():
        rospy.init_node('ggcnn_detector')
        




######## main

def main():
    ggcnn = ros_gg_cnn()
    rospy.sleep(1.0)

	# if args.gazebo:
	# 	raw_input("Move the objects out of the camera view and move the robot to the depth cam shot position before continuing.")
	# 	ggcnn.get_depth_image_shot()

    raw_input("press enter to start GGCNN")
    rospy.loginfo("start ggcnn")
    rate = rospy.Rate(4)

    while not rospy.is_shutdown():
        if args.gazebo:
			number_of_boxes = ggcnn.copy_obj_to_depth_img()
        if number_of_boxes > 0:
            with TimeIt('ggcnn process time lapsed')
                ggcnn.depth_process_ggcnn()
                ggcnn.get_grasp_image()
                ggcnn.publish_images()
                ggcnn.publish_data_to_robot()
        rate.sleep()

if __name__ = '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Program interrupted before completion")