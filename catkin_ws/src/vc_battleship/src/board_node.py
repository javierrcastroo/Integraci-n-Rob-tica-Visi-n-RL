#!/usr/bin/env python
import json
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from board_config import BOARD_CAMERA_PARAMS_PATH, USE_UNDISTORT_BOARD, WARP_SIZE
import aruco_util
import board_ui
from board_runtime import BoardRuntime


class BoardNode(object):
    """Nodo ROS que replica el comportamiento de board_main para el tablero."""

    def __init__(self):
        rospy.init_node("board_node", anonymous=True)

        self.frame_rate = rospy.get_param("~fps", 30.0)
        self.aruco_id = rospy.get_param("~aruco_id", aruco_util.ARUCO_ORIGIN_ID)
        self.objects_topic = rospy.get_param(
            "~objects_topic", "/board/object_states"
        )
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.pub_objects = rospy.Publisher(self.objects_topic, String, queue_size=10)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_frame_stamp = None
        self.last_payload = {"objects": [], "ammo": {"list": [], "selected": None}}
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        rospy.loginfo(
            f"[BOARD] Suscrito a {self.image_topic} para recibir frames del tablero"
        )

        self.cam_mtx = None
        self.dist_coeffs = None
        self.new_cam_mtx = None
        self.map1 = None
        self.map2 = None
        if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
            data = np.load(BOARD_CAMERA_PARAMS_PATH)
            self.cam_mtx = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            rospy.loginfo("[BOARD] Undistort activado para la cámara del tablero")
        else:
            rospy.loginfo("[BOARD] Sin parámetros de calibración o undistort desactivado")

        self.runtime = BoardRuntime(max_boards=2, warp_size=WARP_SIZE, aruco_id=self.aruco_id)

        cv2.namedWindow("Tablero ROS")
        cv2.setMouseCallback("Tablero ROS", board_ui.board_mouse_callback)

        rospy.on_shutdown(self.shutdown)

        service_name = rospy.get_param("~capture_service", "/board/capture_state")
        self.capture_srv = rospy.Service(service_name, Trigger, self.handle_capture_request)
        rospy.loginfo(f"[BOARD] Servicio de captura disponible en {service_name}")

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            if self.latest_frame is None:
                self.show_waiting_screen()
                rospy.logwarn_throttle(
                    5.0,
                    f"[BOARD] Esperando imágenes del tópico {self.image_topic}",
                )
                rate.sleep()
                continue

            frame = self.latest_frame.copy()

            frame_proc = self.apply_undistort(frame)
            frame_proc = cv2.flip(frame_proc, 1)

            try:
                vis, mask_board, obj_mask, objects_info, ammo_data = self.runtime.process_frame(
                    frame_proc
                )
            except Exception as exc:
                rospy.logwarn_throttle(
                    5.0, f"[BOARD] Error procesando el tablero: {exc}"
                )
                rate.sleep()
                continue

            self.publish_objects(objects_info, ammo_data)
            BoardRuntime.draw_origin_indicator(vis)

            cv2.imshow("Tablero ROS", vis)
            if mask_board is not None:
                cv2.imshow("Mascara tablero", mask_board)
            if obj_mask is not None:
                cv2.imshow("Mascara objetos", obj_mask)
            ammo_mask = ammo_data.get("mask") if ammo_data else None
            if ammo_mask is not None:
                cv2.imshow("Mascara municion", ammo_mask)

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
    def publish_objects(self, objects_info, ammo_data):
        if objects_info is None:
            objects_info = []
        if ammo_data is None:
            ammo_data = {"list": [], "selected": None}
        payload = {"objects": objects_info, "ammo": {
            "list": ammo_data.get("list", []),
            "selected": ammo_data.get("selected"),
        }}
        self.last_payload = payload
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_objects.publish(msg)

    def handle_capture_request(self, _req):
        if self.latest_frame is None:
            return TriggerResponse(success=False, message="[]")
        try:
            payload = json.dumps(self.last_payload)
        except Exception as exc:
            rospy.logwarn(f"[BOARD] Error serializando objetos: {exc}")
            return TriggerResponse(success=False, message=str(exc))
        return TriggerResponse(success=True, message=payload)

    # ------------------------------------------------------------------
    # Gestión de teclado (idéntica a board_main pero con logs ROS)
    # ------------------------------------------------------------------
    def handle_keys(self, key, frame):
        if key == ord("b"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = self.runtime.calibrate_board_color(roi_hsv)
                rospy.loginfo(f"[BOARD] Calibrado TABLERO: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI del tablero antes de pulsar 'b'")

        elif key == ord("o"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = self.runtime.calibrate_object_color(roi_hsv)
                rospy.loginfo(f"[BOARD] Calibrado OBJETO: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI sobre la ficha antes de pulsar 'o'")

        elif key == ord("1"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = self.runtime.calibrate_ship_color("ship1", roi_hsv)
                rospy.loginfo(f"[BOARD] Calibrado BARCO x1: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI del barco tamaño 1")

        elif key == ord("2"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = self.runtime.calibrate_ship_color("ship2", roi_hsv)
                rospy.loginfo(f"[BOARD] Calibrado BARCO x2: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI del barco tamaño 2")

        elif key == ord("m"):
            if board_ui.board_roi_defined:
                x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
                y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = self.runtime.calibrate_ammo_color(roi_hsv)
                rospy.loginfo(f"[BOARD] Calibrada MUNICIÓN: {lo} {up}")
            else:
                rospy.logwarn("[BOARD] Dibuja un ROI sobre la munición antes de pulsar 'm'")

        elif key == ord("r"):
            self.runtime.reset_origin()
            rospy.loginfo(
                f"[BOARD] Origen global reiniciado. Esperando ArUco ID {self.aruco_id}"
            )

    # Callback de imagen
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, f"[BOARD] Error al convertir imagen: {exc}")
            return

        self.latest_frame = frame
        self.last_frame_stamp = msg.header.stamp
        rospy.loginfo_once("[BOARD] Recibido primer frame desde la cámara ROS")

        if (
            self.cam_mtx is not None
            and self.dist_coeffs is not None
            and self.map1 is None
            and self.map2 is None
        ):
            self._init_undistort_maps(frame.shape[1], frame.shape[0])

    # ------------------------------------------------------------------
    # Pantalla de espera
    # ------------------------------------------------------------------
    def show_waiting_screen(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Esperando frames...",
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Tablero ROS", placeholder)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Shutdown limpio
    # ------------------------------------------------------------------
    def shutdown(self):
        cv2.destroyAllWindows()
        rospy.loginfo("[BOARD] Nodo cerrado")

    def apply_undistort(self, frame):
        if (
            self.cam_mtx is None
            or self.dist_coeffs is None
            or self.map1 is None
            or self.map2 is None
        ):
            return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _init_undistort_maps(self, width, height):
        self.new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.cam_mtx, self.dist_coeffs, (width, height), 0
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.cam_mtx,
            self.dist_coeffs,
            None,
            self.new_cam_mtx,
            (width, height),
            cv2.CV_16SC2,
        )
        rospy.loginfo(
            "[BOARD] Mapas de undistort preparados para %dx%d"
            % (width, height)
        )


if __name__ == "__main__":
    node = BoardNode()
    try:
        node.spin()
    finally:
        node.shutdown()
