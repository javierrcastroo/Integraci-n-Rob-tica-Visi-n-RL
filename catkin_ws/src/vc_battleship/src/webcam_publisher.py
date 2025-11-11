#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():
    rospy.init_node('webcam_publisher', anonymous=True)

    cap = cv2.VideoCapture(0)  # webcam del portátil
    if not cap.isOpened():
        rospy.logerr("No se pudo abrir la webcam (índice 0)")
        return

    pub = rospy.Publisher('/webcam/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("No se pudo leer frame de la webcam")
            rate.sleep()
            continue

        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "webcam_frame"
        pub.publish(msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
