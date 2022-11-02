import cv2
import mediapipe as mp

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RecogMediaPipe:
    def __init__(self, is_static_mode=False):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_objectron = mp.solutions.objectron
        self.is_static_mode = is_static_mode

        self.bridge = CvBridge()
        rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self.cb_recog)
        self._pub_result = rospy.Publisher("/mediapipe/result/Image", Image, queue_size=10)

    def cb_recog(self, data):

        with self.mp_objectron.Objectron(static_image_mode=self.is_static_mode,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Chair') as objectron:

            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            if self.is_static_mode==False:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = objectron.process(image)

                # Draw the box landmarks on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        self.mp_drawing.draw_landmarks(
                        image, detected_object.landmarks_2d, self.mp_objectron.BOX_CONNECTIONS)
                        self.mp_drawing.draw_axis(image, detected_object.rotation,
                                            detected_object.translation)
                # Flip the image horizontally for a selfie-view display.

                debug_mode = False
                if debug_mode:
                    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
                    cv2.waitKey(0)

            result_smi = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self._pub_result.publish(result_smi)

if __name__ == "__main__":
    rospy.init_node('mediapipe_node')
    recog_mediapipe = RecogMediaPipe()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rate.sleep()