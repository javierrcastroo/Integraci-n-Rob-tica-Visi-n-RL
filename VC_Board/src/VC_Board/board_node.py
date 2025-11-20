#!/usr/bin/env python3
import os
import json
from collections import Counter, defaultdict
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
import board_state
import board_processing as bp
import aruco_utils
import battleship_logic


class LayoutAccumulator:
    def __init__(self, target_frames):
        self.target_frames = target_frames
        self.reset()

    def reset(self):
        self.frame_count = 0
        self.data = defaultdict(
            lambda: {
                "board_size": None,
                "ship_two_counts": Counter(),
                "ship_one_counts": Counter(),
                "ship_two_offsets": defaultdict(list),
                "ship_one_offsets": defaultdict(list),
                "ship_two_pixels": defaultdict(list),
                "ship_one_pixels": defaultdict(list),
            }
        )

    def push(self, layouts):
        self.frame_count += 1
        for layout in layouts:
            name = layout.get("name", f"board_{len(self.data)}")
            entry = self.data[name]

            if layout.get("board_size") is not None:
                entry["board_size"] = layout["board_size"]

            for cell in layout.get("ship_two_cells", []):
                entry["ship_two_counts"][tuple(cell)] += 1
            for cell in layout.get("ship_one_cells", []):
                entry["ship_one_counts"][tuple(cell)] += 1

            for det in layout.get("ship_two_detections", []):
                cell = det.get("cell")
                if cell is None:
                    continue
                cell = tuple(cell)
                pixel = det.get("pixel")
                offset = det.get("offset_from_origin")
                if pixel is not None:
                    entry["ship_two_pixels"][cell].append(tuple(pixel))
                if offset is not None:
                    entry["ship_two_offsets"][cell].append(tuple(offset))

            for det in layout.get("ship_one_detections", []):
                cell = det.get("cell")
                if cell is None:
                    continue
                cell = tuple(cell)
                pixel = det.get("pixel")
                offset = det.get("offset_from_origin")
                if pixel is not None:
                    entry["ship_one_pixels"][cell].append(tuple(pixel))
                if offset is not None:
                    entry["ship_one_offsets"][cell].append(tuple(offset))

    def progress(self):
        if self.target_frames <= 0:
            return 1.0
        return min(1.0, float(self.frame_count) / float(self.target_frames))

    def ready(self):
        return self.frame_count >= self.target_frames

    def _average_point(self, pts):
        if not pts:
            return None
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        n = float(len(pts))
        return (sx / n, sy / n)

    def _cells_with_type(self, ship_two_cells, ship_one_cells):
        cells = []
        for r, c in ship_two_cells:
            cells.append({"row": r, "col": c, "type": "ship_two"})
        for r, c in ship_one_cells:
            cells.append({"row": r, "col": c, "type": "ship_one"})
        return cells

    def build_layouts(self):
        layouts = []
        threshold = max(1, int(self.target_frames * 0.6))

        for name, entry in self.data.items():
            ship_two_cells = [
                cell for cell, count in entry["ship_two_counts"].items() if count >= threshold
            ]
            ship_one_cells = [
                cell for cell, count in entry["ship_one_counts"].items() if count >= threshold
            ]

            ship_two_positions = []
            for cell in ship_two_cells:
                ship_two_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_two_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(
                            entry["ship_two_offsets"].get(cell, [])
                        ),
                    }
                )

            ship_one_positions = []
            for cell in ship_one_cells:
                ship_one_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_one_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(
                            entry["ship_one_offsets"].get(cell, [])
                        ),
                    }
                )

            layout = {
                "name": name,
                "board_size": entry["board_size"],
                "ship_two_cells": sorted(ship_two_cells),
                "ship_one_cells": sorted(ship_one_cells),
                "cells": self._cells_with_type(ship_two_cells, ship_one_cells),
                "ship_two_positions": ship_two_positions,
                "ship_one_positions": ship_one_positions,
            }
            layouts.append(layout)

        return layouts


class BoardNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.last_frame = None

        self.capture_frames = rospy.get_param("~capture_frames", 150)
        self.capture_state = "WAIT_TRIGGER"
        self.capture_progress = 0.0
        self.status_lines = ["Ajusta HSV y pulsa 's' para enviar el tablero."]
        self.accumulator = None
        self.capture_reason = "manual"

        image_topic = rospy.get_param("~image_topic", "board_camera/image_raw")
        rospy.loginfo(f"[board_node] Suscribiéndose a: {image_topic}")
        self.sub = rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)

        # publicador de layout de tablero y subscripción a peticiones
        self.board_pub = rospy.Publisher("battleship/board_layout", String, queue_size=10)
        self.request_sub = rospy.Subscriber(
            "battleship/board_request", String, self.request_cb, queue_size=10
        )

        # calibración
        self.mtx = None
        self.dist = None
        if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
            data = np.load(BOARD_CAMERA_PARAMS_PATH)
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            rospy.loginfo("[board_node] Undistort activado para tablero")

        # estado de tableros
        self.boards_state_list = [
            board_state.init_board_state("T1"),
            board_state.init_board_state("T2"),
        ]

        # ventanas
        cv2.namedWindow("Tablero")
        cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)
        cv2.namedWindow("Mascara tablero")
        cv2.namedWindow("Mascara barco x2")
        cv2.namedWindow("Mascara barco x1")
        cv2.namedWindow("Mascara municion")

        # timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.timer_cb)

    def cb_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"[board_node] Error CvBridge: {e}")
            return
        self.last_frame = frame

    def timer_cb(self, event):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        # undistort
        if self.mtx is not None and self.dist is not None:
            frame = cv2.undistort(frame, self.mtx, self.dist)

        # origen ArUco
        aruco_utils.update_global_origin_from_aruco(frame, aruco_id=2)

        # process boards
        vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_all_boards(
            frame,
            self.boards_state_list,
            cam_mtx=self.mtx,
            dist=self.dist,
            max_boards=2,
            warp_size=WARP_SIZE,
        )

        # validación de cada layout (como antes)
        validation_map = {}
        for layout in layouts:
            ok, msg_text = battleship_logic.evaluate_board(layout)
            validation_map[layout["name"]] = (ok, msg_text)
            print(f"[{layout['name']}] {msg_text}")

        for slot in self.boards_state_list:
            if slot["name"] in validation_map and slot["last_quad"] is not None:
                ok, msg_text = validation_map[slot["name"]]
                board_ui.draw_validation_result(vis, slot["last_quad"], msg_text, ok)

        # origen global
        if board_state.GLOBAL_ORIGIN is not None:
            gx, gy = board_state.GLOBAL_ORIGIN
            cv2.circle(vis, (int(gx), int(gy)), 10, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "ORIGEN (ArUco)",
                (int(gx) + 10, int(gy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        self.update_capture_state(layouts)

        board_ui.draw_board_hud(vis)
        board_ui.draw_capture_status(
            vis, self.capture_state, self.capture_progress, self.status_lines
        )

        # mostrar
        cv2.imshow("Tablero", vis)
        if mask_b is not None:
            cv2.imshow("Mascara tablero", mask_b)
        if mask_ship2 is not None:
            cv2.imshow("Mascara barco x2", mask_ship2)
        if mask_ship1 is not None:
            cv2.imshow("Mascara barco x1", mask_ship1)
        if mask_m is not None:
            cv2.imshow("Mascara municion", mask_m)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            rospy.loginfo("[board_node] Saliendo por ESC/q")
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User exit")
            return

        self.handle_keys(key, frame)

    def start_capture(self, reason):
        if self.capture_state == "CAPTURING":
            rospy.loginfo("[board_node] Captura ya en curso, petición ignorada")
            return

        rospy.loginfo(f"[board_node] Iniciando captura de layout ({reason})")
        self.capture_reason = reason
        self.accumulator = LayoutAccumulator(self.capture_frames)
        self.capture_state = "CAPTURING"
        self.capture_progress = 0.0
        self.status_lines = [
            f"Capturando {self.capture_frames} frames ({reason}).",
            "No muevas el tablero.",
        ]

    def publish_layouts(self, layouts):
        payload = {"boards": layouts}
        msg = String()
        msg.data = json.dumps(payload, default=self.json_default)
        self.board_pub.publish(msg)
        rospy.loginfo(
            f"[board_node] Layouts enviados ({len(layouts)} tableros) tras captura."
        )

    def update_capture_state(self, layouts):
        if self.capture_state == "CAPTURING":
            if self.accumulator is None:
                self.accumulator = LayoutAccumulator(self.capture_frames)
            self.accumulator.push(layouts)
            self.capture_progress = self.accumulator.progress()

            if self.accumulator.ready():
                averaged = self.accumulator.build_layouts()
                if averaged:
                    self.publish_layouts(averaged)
                    self.status_lines = [
                        f"Layout enviado ({self.capture_reason}).",
                        "Standby: esperando peticiones automáticas.",
                    ]
                else:
                    self.status_lines = [
                        "No se detectó un tablero estable durante la captura.",
                        "Repite la operación cuando haya imagen.",
                    ]
                self.capture_state = "STANDBY"
        elif self.capture_state == "STANDBY":
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Standby: esperando petición del juego."]
        else:
            # WAIT_TRIGGER u otros estados
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Ajusta HSV y pulsa 's' para enviar el tablero."]

    def request_cb(self, msg):
        reason = msg.data if msg and msg.data else "petición automática"
        rospy.loginfo(f"[board_node] Petición externa de layout: {reason}")
        if self.capture_state == "WAIT_TRIGGER":
            self.status_lines = ["Petición recibida, iniciando captura de tablero."]
        self.start_capture(reason)

    @staticmethod
    def json_default(o):
        # para convertir tuples a listas si aparecen en layouts
        if isinstance(o, tuple):
            return list(o)
        raise TypeError

    def handle_keys(self, key, frame):
        import board_tracker
        import object_tracker
        import board_ui as bu

        if key == ord("b"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
                board_tracker.current_lower, board_tracker.current_upper = lo, up
                print("[INFO] calibrado TABLERO:", lo, up)
            else:
                print("[WARN] dibuja ROI en 'Tablero' primero")

        elif key == ord("2"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ship_two_color_from_roi(roi_hsv)
                object_tracker.current_ship_two_lower, object_tracker.current_ship_two_upper = lo, up
                print("[INFO] calibrado BARCO x2:", lo, up)
            else:
                print("[WARN] dibuja ROI sobre el barco largo")

        elif key == ord("1"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ship_one_color_from_roi(roi_hsv)
                object_tracker.current_ship_one_lower, object_tracker.current_ship_one_upper = lo, up
                print("[INFO] calibrado BARCO x1:", lo, up)
            else:
                print("[WARN] dibuja ROI sobre el barco corto")

        elif key == ord("m"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ammo_color_from_roi(roi_hsv)
                object_tracker.current_ammo_lower, object_tracker.current_ammo_upper = lo, up
                print("[INFO] calibrada MUNICION:", lo, up)
            else:
                print("[WARN] dibuja ROI sobre la municion")

        elif key == ord("s"):
            self.start_capture("tecla 's'")


def main():
    rospy.init_node("board_node", anonymous=True)
    node = BoardNode()
    rospy.loginfo("[board_node] Nodo de tablero iniciado.")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
