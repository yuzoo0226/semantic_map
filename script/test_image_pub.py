#!/usr/bin/env python3
import cv2
import rospy
import sys

import roslib
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

sys.path.append(roslib.packages.get_pkg_dir("semantic_map") + "/include/omni3d/")


class TestImagePubliher():
    def __init__(self) -> None:
        image_path = roslib.packages.get_pkg_dir("semantic_map") + "/io/images/chair.jpg"
        self.test_image = cv2.imread(image_path, 1)
        self.pub_image = rospy.Publisher("/image/test", Image, queue_size=10)
        self.bridge = CvBridge()

        self.smi = self.bridge.cv2_to_imgmsg(self.test_image, encoding="bgr8")

    def publish_image(self):
        self.pub_image.publish(self.smi)


if __name__ == "__main__":
    rospy.init_node("test_image_publisher")
    im_pub = TestImagePubliher()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        im_pub.publish_image()
        rate.sleep


