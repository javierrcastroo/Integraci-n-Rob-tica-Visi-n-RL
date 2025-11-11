#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():
    rospy.init_node('usb_cam_publisher', anonymous=True)

    cap = cv2.VideoCapture(1)  # cámara USB
    if not cap.isOpened():
        rospy.logerr("No se pudo abrir la cámara USB (índice 1)")
        return

    pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("No se pudo leer frame de la cámara USB")
            rate.sleep()
            continue

        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "usb_cam_frame"
        pub.publish(msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
