#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def try_open_camera(indices):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            rospy.loginfo("[webcam_publisher] Usando cámara USB secundaria con índice %d", idx)
            return cap, idx
        cap.release()
        rospy.logwarn("[webcam_publisher] No se pudo abrir la cámara secundaria con índice %d", idx)
    return None, None


def main():
    rospy.init_node('webcam_publisher', anonymous=True)

    preferred = rospy.get_param('~preferred_indices', [1, 3, 5, 7, 0, 2, 4, 6])
    if isinstance(preferred, int):
        preferred = [preferred]

    cap, index = try_open_camera(preferred)
    if cap is None:
        rospy.logerr("[webcam_publisher] No se pudo abrir ninguna cámara secundaria con los índices %s", preferred)
        return

    pub = rospy.Publisher('/webcam/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("No se pudo leer frame de la cámara USB secundaria")
            rate.sleep()
            continue

        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = f"usb_cam_secondary_frame_{index}"
        pub.publish(msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
