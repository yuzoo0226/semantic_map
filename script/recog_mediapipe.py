#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Mapping, Optional, Tuple, Union

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters


class RecogMediaPipe:
    def __init__(self, is_static_mode=False):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_objectron = mp.solutions.objectron
        self.is_static_mode = is_static_mode

        self.bridge = CvBridge()

        # ROS interface

        # topic name
        self._p_rgb = "/hsrb/head_rgbd_sensor/rgb/image_raw"
        self._p_d = "/hsrb/head_rgbd_sensor/depth_registered/image_raw"

        self._sub_smi_rgb = message_filters.Subscriber(self._p_rgb, Image)
        self._sub_smi_d = message_filters.Subscriber(self._p_d, Image)
        interface = [self._sub_smi_rgb, self._sub_smi_d]

        self._sync = message_filters.ApproximateTimeSynchronizer(interface, 5, 0.05, allow_headerless = True)
        self._sync.registerCallback(self.cb_recog)

        # rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self.cb_recog)
        self._pub_result = rospy.Publisher("/mediapipe/result/Image", Image, queue_size=10)


    def get_axis(
        self,
        image: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        focal_length: Tuple[float, float] = (1.0, 1.0),
        principal_point: Tuple[float, float] = (0.0, 0.0),
        axis_length: float = 0.1,):
        # axis_drawing_spec: DrawingSpec = DrawingSpec()):
            """Draws the 3D axis on the image.
            Args:
                image: A three channel BGR image represented as numpy ndarray.
                rotation: Rotation matrix from object to camera coordinate frame.
                translation: Translation vector from object to camera coordinate frame.
                focal_length: camera focal length along x and y directions.
                principal_point: camera principal point in x and y.
                axis_length: length of the axis in the drawing.
                axis_drawing_spec: A DrawingSpec object that specifies the xyz axis
                drawing settings such as line thickness.
            Raises:
                ValueError: If one of the followings:
                a) If the input image is not three channel BGR.
            """
            _BGR_CHANNELS = 3
            if image.shape[2] != _BGR_CHANNELS:
                raise ValueError('Input image must contain three channel bgr data.')
            image_rows, image_cols, _ = image.shape
            # Create axis points in camera coordinate frame.
            axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
            x = axis_cam[..., 0]
            y = axis_cam[..., 1]
            z = axis_cam[..., 2]
            # Project 3D points to NDC space.
            fx, fy = focal_length
            px, py = principal_point
            x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
            y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
            # Convert from NDC space to image space.
            x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
            y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
            # Draw xyz axis on the image.
            origin = (x_im[0], y_im[0])
            x_axis = (x_im[1], y_im[1])
            y_axis = (x_im[2], y_im[2])
            z_axis = (x_im[3], y_im[3])
            
            show_result = False
            if show_result:
                # cv2.arrowedLine(image, origin, x_axis, RED_COLOR, axis_drawing_spec.thickness)
                # cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                #                 axis_drawing_spec.thickness)
                # cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                #                 axis_drawing_spec.thickness)
                pass

            return (origin, x_axis, y_axis, z_axis)


    def cb_recog(self, smi_rgb, smi_d):

        with self.mp_objectron.Objectron(static_image_mode=self.is_static_mode,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Chair') as objectron:

            image = self.bridge.imgmsg_to_cv2(smi_rgb, "bgr8")

            if self.is_static_mode==False:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = objectron.process(image)

                # Draw the box landmarks on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        # self.mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, self.mp_objectron.BOX_CONNECTIONS)
                        self.mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
                        object_position = self.get_axis(image, detected_object.rotation, detected_object.translation)
                        print(object_position[0])
                # Flip the image horizontally for a selfie-view display.

                debug_mode = False
                if debug_mode:
                    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
                    cv2.waitKey(0)

            result_smi = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self._pub_result.publish(result_smi)


    # def calc_3d_array(self, ):

if __name__ == "__main__":
    rospy.init_node('mediapipe_node')
    recog_mediapipe = RecogMediaPipe()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rate.sleep()