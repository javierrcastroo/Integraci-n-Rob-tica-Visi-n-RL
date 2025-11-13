#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def try_open_camera(indices):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            rospy.loginfo("[usb_cam_publisher] Usando cámara USB con índice %d", idx)
            return cap, idx
        cap.release()
        rospy.logwarn("[usb_cam_publisher] No se pudo abrir la cámara con índice %d", idx)
    return None, None


def main():
    rospy.init_node('usb_cam_publisher', anonymous=True)

    preferred = rospy.get_param('~preferred_indices', [0, 2, 4, 6, 1, 3, 5, 7])
    if isinstance(preferred, int):
        preferred = [preferred]

    cap, index = try_open_camera(preferred)
    if cap is None:
        rospy.logerr("[usb_cam_publisher] No se pudo abrir ninguna cámara con los índices %s", preferred)
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
        msg.header.frame_id = f"usb_cam_frame_{index}"
        pub.publish(msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
