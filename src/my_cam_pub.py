#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node("my_cam_pub", anonymous = True)
cap = cv2.VideoCapture(0 ,cv2.CAP_V4L)

if cap.isOpened():
    pub = rospy.Publisher("com47/my_image", Image, queue_size=10)
    bridge = CvBridge()

    fps = cap.get(cv2.CAP_PROP_FPS)
    loop_rate = rospy.Rate(fps)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(msg)
            loop_rate.sleep()
        except CvBridgeError as e:
            print(e)
        loop_rate.sleep()