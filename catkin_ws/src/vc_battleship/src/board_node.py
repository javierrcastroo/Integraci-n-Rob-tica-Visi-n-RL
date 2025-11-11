#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# importa tus módulos del tablero
import board_state
import board_processing
import aruco_utils  # el que hicimos con DICT_5X5_100

class BoardNode(object):
    def __init__(self):
        rospy.init_node('board_node', anonymous=True)

        self.bridge = CvBridge()
        self.last_frame = None

        # inicializamos los 2 slots como en tu código
        self.boards_state_list = [
            board_state.init_board_state("T1"),
            board_state.init_board_state("T2"),
        ]

        rospy.Subscriber('/usb_cam/image_raw', Image, self.cb_image)

        # timer para procesar a ~15 fps (no hace falta más)
        self.timer = rospy.Timer(rospy.Duration(1.0/15.0), self.timer_cb)

    def cb_image(self, msg):
        # guardamos el último frame en BGR
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def timer_cb(self, event):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        # 1) actualizar origen ArUco en este frame
        # (si no hay ArUco, el módulo ya mantiene el último durante unos frames)
        aruco_utils.update_global_origin_from_aruco(frame, aruco_id=0)

        # 2) procesar todos los tableros
        vis, mask_board, obj_mask, _ = board_processing.process_all_boards(
            frame,
            self.boards_state_list,
            cam_mtx=None,
            dist=None,
            max_boards=2,
            warp_size=500
        )

        # mostrar para debug (como en tu script original)
        cv2.imshow("Tableros ROS", vis)
        if mask_board is not None:
            cv2.imshow("Mask tablero", mask_board)
        if obj_mask is not None:
            cv2.imshow("Mask objetos", obj_mask)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = BoardNode()
    rospy.spin()
    cv2.destroyAllWindows()
