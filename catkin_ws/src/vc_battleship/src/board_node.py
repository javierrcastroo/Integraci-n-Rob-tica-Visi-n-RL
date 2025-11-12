#!/usr/bin/env python
import json
import os

import cv2
import numpy as np
import rospy
from std_msgs.msg import String

from board_config import BOARD_CAMERA_PARAMS_PATH, USE_UNDISTORT_BOARD, WARP_SIZE
import aruco_util
import board_processing as bp
import board_state
import board_tracker
import board_ui
import object_tracker


class BoardNode(object):
    """Nodo ROS que replica el comportamiento de board_main para el tablero."""

    def __init__(self):
        rospy.init_node("board_node", anonymous=True)

        self.camera_index = rospy.get_param("~camera_index", 1)
        self.frame_rate = rospy.get_param("~fps", 30.0)
        self.aruco_id = rospy.get_param("~aruco_id", aruco_util.ARUCO_ORIGIN_ID)
        topic = rospy.get_param("~objects_topic", "/board/object_states")
        self.pub_objects = rospy.Publisher(topic, String, queue_size=10)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[BOARD] No se pudo abrir la cámara {self.camera_index}"
            )

        self.cam_mtx = None
        self.dist_coeffs = None
        if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
            data = np.load(BOARD_CAMERA_PARAMS_PATH)
            self.cam_mtx = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            rospy.loginfo("[BOARD] Undistort activado para la cámara del tablero")
        else:
            rospy.loginfo("[BOARD] Sin parámetros de calibración o undistort desactivado")

        self.boards_state = [
            board_state.init_board_state("T1"),
            board_state.init_board_state("T2"),
        ]

        cv2.namedWindow("Tablero ROS")
        cv2.setMouseCallback("Tablero ROS", board_ui.board_mouse_callback)

        rospy.on_shutdown(self.shutdown)

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if not ok:
                rospy.logwarn_throttle(5.0, "[BOARD] Frame inválido de la cámara")
                rate.sleep()
                continue

            frame_proc = frame
            if self.cam_mtx is not None and self.dist_coeffs is not None:
                frame_proc = cv2.undistort(frame, self.cam_mtx, self.dist_coeffs)

            # actualizar el origen global mediante ArUco
            try:
                aruco_util.update_global_origin_from_aruco(
                    frame_proc, aruco_id=self.aruco_id
                )
            except Exception as exc:
                rospy.logwarn_throttle(
                    5.0, f"[BOARD] Error actualizando origen ArUco: {exc}"
                )

            vis, mask_board, obj_mask, objects_info = bp.process_all_boards(
                frame_proc,
                self.boards_state,
                cam_mtx=self.cam_mtx,
                dist=self.dist_coeffs,
                max_boards=2,
                warp_size=WARP_SIZE,
            )

            self.publish_objects(objects_info)
            self.draw_origin_indicator(vis)

            cv2.imshow("Tablero ROS", vis)
            if mask_board is not None:
                cv2.imshow("Mascara tablero", mask_board)
            if obj_mask is not None:
                cv2.imshow("Mascara objetos", obj_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("Salida solicitada por el usuario")
                break

            if key != 255:
                self.handle_keys(key, frame_proc)

            rate.sleep()

    # ------------------------------------------------------------------
    # Publicación hacia game_logic
    # ------------------------------------------------------------------
    def publish_objects(self, objects_info):
        if objects_info is None:
            return
        msg = String()
        msg.data = json.dumps(objects_info)
        self.pub_objects.publish(msg)

    # ------------------------------------------------------------------
    # Gestión de teclado (idéntica a board_main pero con logs ROS)
    # ------------------------------------------------------------------
    def handle_keys(self, key, frame):
        if key == ord("b"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
                board_tracker.current_lower, board_tracker.current_upper = lo, up
                rospy.loginfo(f"[BOARD] Calibrado TABLERO: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI del tablero antes de pulsar 'b'")

        elif key == ord("o"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_object_color_from_roi(roi_hsv)
                object_tracker.current_obj_lower, object_tracker.current_obj_upper = lo, up
                rospy.loginfo(f"[BOARD] Calibrado OBJETO: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI sobre la ficha antes de pulsar 'o'")

        elif key == ord("r"):
            board_state.GLOBAL_ORIGIN = None
            board_state.GLOBAL_ORIGIN_MISS = board_state.GLOBAL_ORIGIN_MAX_MISS + 1
            rospy.loginfo(
                f"[BOARD] Origen global reiniciado. Esperando ArUco ID {self.aruco_id}"
            )

    # ------------------------------------------------------------------
    # Dibujar origen global
    # ------------------------------------------------------------------
    def draw_origin_indicator(self, vis_img):
        if board_state.GLOBAL_ORIGIN is None:
            return
        gx, gy = board_state.GLOBAL_ORIGIN
        cv2.circle(vis_img, (int(gx), int(gy)), 10, (0, 255, 0), -1)
        cv2.putText(
            vis_img,
            "ORIGEN (ArUco)",
            (int(gx) + 10, int(gy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ------------------------------------------------------------------
    # Shutdown limpio
    # ------------------------------------------------------------------
    def shutdown(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("[BOARD] Nodo cerrado")


if __name__ == "__main__":
    node = BoardNode()
    try:
        node.spin()
    finally:
        node.shutdown()
