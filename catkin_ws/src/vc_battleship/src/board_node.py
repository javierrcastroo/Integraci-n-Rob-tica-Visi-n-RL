#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import board_state
import board_processing
import aruco_utils


class BoardNode(object):
    def __init__(self):
        rospy.init_node('board_node', anonymous=True)

        self.bridge = CvBridge()
        self.last_frame = None

        self.boards_state_list = [
            board_state.init_board_state("T1"),
            board_state.init_board_state("T2"),
        ]

        # HSV por defecto
        self.tablero_hsv = ((30, 50, 50), (90, 255, 255))
        self.objetos_hsv = ((0, 50, 50), (179, 255, 255))
        self.selection_mode = "board"

        # para el drag
        self.dragging = False
        self.start_pt = (0, 0)
        self.end_pt = (0, 0)

        # buffers de visualización (los rellenará el timer)
        self.vis_img = None
        self.mask_board_img = None
        self.obj_mask_img = None
        self.debug_img = None

        rospy.Subscriber('/usb_cam/image_raw', Image, self.cb_image)

        # timer de procesamiento
        self.timer = rospy.Timer(rospy.Duration(1.0/15.0), self.timer_cb)

        # aruco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.target_aruco_id = rospy.get_param("~aruco_id", 0)

        # ventanas + ratón
        cv2.namedWindow("Tableros ROS")
        cv2.setMouseCallback("Tableros ROS", self.on_mouse)

    # ===== ROS =====
    def cb_image(self, msg):
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # ===== ratón =====
    def on_mouse(self, event, x, y, flags, param):
        if self.last_frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end_pt = (x, y)
            self.extract_hsv_from_roi()

    def extract_hsv_from_roi(self):
        if self.last_frame is None:
            return

        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        if x_max - x_min < 5 or y_max - y_min < 5:
            rospy.logwarn("ROI demasiado pequeño, no actualizo HSV")
            return

        roi_bgr = self.last_frame[y_min:y_max, x_min:x_max]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(roi_hsv)
        h_min, h_max = int(h.min()), int(h.max())
        s_min, s_max = int(s.min()), int(s.max())
        v_min, v_max = int(v.min()), int(v.max())

        # margen
        h_min = max(h_min - 5, 0)
        s_min = max(s_min - 10, 0)
        v_min = max(v_min - 10, 0)
        h_max = min(h_max + 5, 179)
        s_max = min(s_max + 10, 255)
        v_max = min(v_max + 10, 255)

        hsv_range = ((h_min, s_min, v_min), (h_max, s_max, v_max))

        if self.selection_mode == "board":
            self.tablero_hsv = hsv_range
            rospy.loginfo("HSV tablero = %s", str(hsv_range))
        else:
            self.objetos_hsv = hsv_range
            rospy.loginfo("HSV objetos = %s", str(hsv_range))

    # ===== aruco debug =====
    def detect_and_draw_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        seen = False
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_aruco_id:
                    seen = True
        return seen

    # ===== llamada segura al procesamiento =====
    def safe_process_all_boards(self, frame):
        try:
            return board_processing.process_all_boards(
                frame,
                self.boards_state_list,
                cam_mtx=None,
                dist=None,
                max_boards=2,
                warp_size=500,
                tablero_hsv=self.tablero_hsv,
                objetos_hsv=self.objetos_hsv
            )
        except TypeError:
            return board_processing.process_all_boards(
                frame,
                self.boards_state_list,
                cam_mtx=None,
                dist=None,
                max_boards=2,
                warp_size=500
            )

    # ===== timer: SOLO calcula y guarda =====
    def timer_cb(self, event):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        # aruco tuyo
        try:
            aruco_utils.update_global_origin_from_aruco(frame, aruco_id=self.target_aruco_id)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Fallo update_global_origin_from_aruco: %s" % str(e))

        # aruco debug
        debug_frame = frame.copy()
        seen = self.detect_and_draw_aruco(debug_frame)
        if not seen:
            rospy.logwarn_throttle(5.0, "No veo ArUco ID=%d" % self.target_aruco_id)

        vis, mask_board, obj_mask, _ = self.safe_process_all_boards(frame)

        # si estás arrastrando, dibuja el rectángulo
        if self.dragging:
            cv2.rectangle(vis, self.start_pt, self.end_pt, (0, 255, 0), 2)

        # guarda todo en atributos para que el main lo muestre
        self.vis_img = vis
        self.mask_board_img = mask_board
        self.obj_mask_img = obj_mask
        self.debug_img = debug_frame


if __name__ == '__main__':
    node = BoardNode()

    # bucle principal SOLO para mostrar y leer teclas
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # mostrar solo si hay algo
        if node.vis_img is not None:
            cv2.imshow("Tableros ROS", node.vis_img)
        if node.mask_board_img is not None:
            cv2.imshow("Mask tablero", node.mask_board_img)
        if node.obj_mask_img is not None:
            cv2.imshow("Mask objetos", node.obj_mask_img)
        if node.debug_img is not None:
            cv2.imshow("Frame debug", node.debug_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            node.selection_mode = "board"
            rospy.loginfo("Modo selección → TABLERO")
        elif key == ord('o'):
            node.selection_mode = "obj"
            rospy.loginfo("Modo selección → OBJETOS")

        rate.sleep()

    cv2.destroyAllWindows()
