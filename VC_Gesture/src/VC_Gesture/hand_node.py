#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
from collections import deque

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from hand_config import (
    PREVIEW_W, PREVIEW_H,
    RECOGNIZE_MODE,
    CONFIDENCE_THRESHOLD,
    USE_UNDISTORT_HAND,
    HAND_CAMERA_PARAMS_PATH,
)

import ui
from segmentation import (
    calibrate_from_roi,
    segment_hand_mask,
    hsv_medians,
)
from features import compute_feature_vector
from classifier import knn_predict
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json

GESTURE_WINDOW_FRAMES = 150
MAX_SEQUENCE_LENGTH = 2
TRIGGER_GESTURES = {"demond", "demonio"}
CONFIRM_GESTURE = "ok"
REJECT_GESTURE = "nook"
PRINT_GESTURE = "cool"
CONTROL_GESTURES = TRIGGER_GESTURES | {CONFIRM_GESTURE, REJECT_GESTURE, PRINT_GESTURE}


def majority_vote(labels):
    if not labels:
        return None
    return max(set(labels), key=labels.count)


class GestureWindow:
    def __init__(self, size=GESTURE_WINDOW_FRAMES):
        self.size = size
        self.labels = []

    def reset(self):
        self.labels = []

    def push(self, label):
        label = label if label is not None else "????"
        self.labels.append(label)
        if len(self.labels) >= self.size:
            winner = majority_vote(self.labels)
            self.reset()
            return winner
        return None

    def progress(self):
        if self.size == 0:
            return 0.0
        return min(1.0, len(self.labels) / float(self.size))


class HandNode(object):
    def __init__(self):
        # --- ROS init ---
        self.bridge = CvBridge()
        self.last_frame = None

        image_topic = rospy.get_param("~image_topic", "hand_camera/image_raw")
        rospy.loginfo(f"[hand_node] Suscribiéndose a imagen en: {image_topic}")
        self.sub = rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)

        # Publisher de ataque y suscriptor de resultado
        self.attack_pub = rospy.Publisher("battleship/attack", String, queue_size=10)
        self.result_sub = rospy.Subscriber("battleship/attack_result", String, self.result_cb, queue_size=10)

        # --- Undistort cámara mano ---
        self.HAND_CAM_MTX = None
        self.HAND_DIST = None
        if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
            data = np.load(HAND_CAMERA_PARAMS_PATH)
            self.HAND_CAM_MTX = data["camera_matrix"]
            self.HAND_DIST = data["dist_coeffs"]
            rospy.loginfo("[hand_node] Undistort activado para la mano")

        # --- Estado interno ---
        self.lower_skin = None
        self.upper_skin = None
        self.white_ref = None

        self.gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
        self.current_label = "2dedos"
        self.acciones = []
        self.recent_preds = deque(maxlen=7)
        self.capture_state = "STANDBY"
        self.pending_candidate = None
        self.gesture_window = GestureWindow()
        self.status_lines = ["Standby: haz 'demond' para activar el registro."]
        self.last_result_msg = ""

        # Ventanas
        cv2.namedWindow("Mano")
        cv2.setMouseCallback("Mano", ui.mouse_callback)
        cv2.namedWindow("Mascara mano")
        cv2.namedWindow("Solo piel mano")

        # Timer principal
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.timer_cb)

    # ---------- helpers estado ----------
    def set_state(self, new_state, lines):
        self.capture_state = new_state
        self.status_lines = lines
        self.gesture_window.reset()

    def set_status(self, lines):
        self.status_lines = lines

    # ---------- callback imagen ----------
    def cb_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"[hand_node] Error CvBridge: {e}")
            return
        self.last_frame = frame

    # ---------- callback resultado ataque ----------
    def result_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[hand_node] Error parseando resultado ataque: {e}")
            return

        cell = data.get("cell", {})
        cell_name = cell.get("name", "?")
        result = data.get("result", "unknown")
        message = data.get("message", "")

        rospy.loginfo(f"[hand_node] Resultado ataque en {cell_name}: {result} - {message}")
        # actualizamos líneas de estado para que se vean en el HUD
        self.last_result_msg = message or f"Resultado: {result} en {cell_name}"
        self.status_lines = [self.last_result_msg]

    # ---------- bucle principal ----------
    def timer_cb(self, event):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        # undistort
        if self.HAND_CAM_MTX is not None:
            frame = cv2.undistort(frame, self.HAND_CAM_MTX, self.HAND_DIST)

        # espejo + resize
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ROI
        ui.draw_roi_rectangle(vis)

        # segmentar mano
        mask = segment_hand_mask(hsv, self.lower_skin, self.upper_skin)
        ui.draw_hand_box(vis, mask)
        skin_only = cv2.bitwise_and(frame, frame, mask=mask)

        # features
        feat_vec = compute_feature_vector(mask)

        # reconocimiento
        best_dist = None
        per_frame_label = None
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, self.gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

        if per_frame_label is not None:
            self.recent_preds.append(per_frame_label)
        stable_label = majority_vote(list(self.recent_preds))

        # HUD
        ui.draw_hud(
            vis,
            self.lower_skin,
            self.upper_skin,
            self.current_label,
        )
        ui.draw_prediction(vis, stable_label, best_dist if best_dist else 0.0)
        ui.draw_sequence_status(
            vis,
            self.acciones,
            self.capture_state,
            self.pending_candidate,
            self.status_lines,
            self.gesture_window.progress(),
        )

        # mostrar
        cv2.imshow("Mano", vis)
        cv2.imshow("Mascara mano", mask)
        cv2.imshow("Solo piel mano", skin_only)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            rospy.loginfo("[hand_node] Saliendo por tecla ESC/Q")
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User exit")
            return

        # -------- flujo controlado por gestos --------
        resolved_label = self.gesture_window.push(stable_label)

        if resolved_label is not None:
            if self.capture_state == "STANDBY":
                if resolved_label in TRIGGER_GESTURES:
                    self.set_state("CAPTURA", ["Sistema activo: muestra el primer gesto."])
                else:
                    self.set_status(["Sigue en standby, haz 'demond' para comenzar."])

            elif self.capture_state == "CAPTURA":
                if resolved_label == "????" or resolved_label in CONTROL_GESTURES:
                    self.set_status(["Gesto no válido, repítelo."])
                else:
                    self.pending_candidate = resolved_label
                    self.set_state(
                        "CONFIRMACION",
                        [
                            f"¿Tu gesto es '{self.pending_candidate}'?",
                            "Confirma con 'ok' o repite con 'nook'.",
                        ],
                    )

            elif self.capture_state == "CONFIRMACION":
                if resolved_label == CONFIRM_GESTURE and self.pending_candidate:
                    self.acciones.append(self.pending_candidate)
                    rospy.loginfo(f"[hand_node] Añadido gesto confirmado: {self.pending_candidate}")
                    self.pending_candidate = None
                    if len(self.acciones) >= MAX_SEQUENCE_LENGTH:
                        self.set_state(
                            "COOL",
                            ["Secuencia completa, haz 'cool' para lanzar el ataque."],
                        )
                    else:
                        self.set_state("CAPTURA", ["Gesto guardado. Muestra el siguiente gesto."])
                elif resolved_label == REJECT_GESTURE:
                    rospy.loginfo("[hand_node] Gesto rechazado, repite el anterior.")
                    self.pending_candidate = None
                    self.set_state("CAPTURA", ["Repite el gesto a registrar."])
                else:
                    self.set_status(["Se esperaba 'ok' o 'nook'."])

            elif self.capture_state == "COOL":
                if resolved_label == PRINT_GESTURE and len(self.acciones) == MAX_SEQUENCE_LENGTH:
                    # Aquí lanzamos el ataque
                    self.send_attack(self.acciones)
                    # guardamos también en JSON (comportamiento original)
                    print("[INFO] Secuencia final:", self.acciones)
                    save_sequence_json(self.acciones)

                    # limpiamos
                    self.acciones.clear()
                    self.pending_candidate = None
                    self.set_state(
                        "STANDBY",
                        ["Standby: haz 'demond' para activar un nuevo registro."],
                    )
                else:
                    self.set_status(["Secuencia lista. Usa 'cool' para lanzar el ataque."])

        # -------- teclas de mano (calibración, guardado, etc.) --------
        if key == ord('c'):
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    self.lower_skin, self.upper_skin = calibrate_from_roi(roi_hsv)
                    rospy.loginfo(f"[hand_node] calibrado HSV mano: {self.lower_skin}, {self.upper_skin}")
                else:
                    rospy.logwarn("[hand_node] ROI muy pequeño")
            else:
                rospy.logwarn("[hand_node] Dibuja un ROI en 'Mano' primero")

        elif key == ord('b'):
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    self.white_ref = {
                        "median": hsv_medians(roi_hsv),
                        "roi": (x0, x1, y0, y1),
                    }
                    rospy.loginfo(f"[hand_node] calibrado blanco ref en ROI: {self.white_ref['median']}")
                else:
                    rospy.logwarn("[hand_node] ROI muy pequeño para referencia blanca")
            else:
                rospy.logwarn("[hand_node] Dibuja un ROI en 'Mano' primero")

        elif key == ord('g'):
            if feat_vec is not None:
                save_gesture_example(feat_vec, self.current_label)
                if RECOGNIZE_MODE:
                    self.gallery.append((feat_vec, self.current_label))
                rospy.loginfo(f"[hand_node] guardado gesto {self.current_label}")
            else:
                rospy.logwarn("[hand_node] no hay gesto válido")

        elif key in (
            ord('0'),
            ord('1'),
            ord('2'),
            ord('3'),
            ord('4'),
            ord('5'),
            ord('d'),
            ord('p'),
            ord('-'),
            ord('n'),
        ):
            mapping = {
                ord('0'): "0dedos",
                ord('1'): "1dedo",
                ord('2'): "2dedos",
                ord('3'): "3dedos",
                ord('4'): "4dedos",
                ord('5'): "5dedos",
                ord('d'): "demonio",
                ord('p'): "ok",
                ord('-'): "cool",
                ord('n'): "nook",
            }
            self.current_label = mapping[key]

    # ---------- envío del ataque al game_logic_node ----------
    def send_attack(self, acciones):
        # acciones es algo como ["0dedos", "4dedos"]
        payload = {
            "player": "P1",
            "gestures": list(acciones),
        }
        msg = String()
        msg.data = json.dumps(payload)
        rospy.loginfo(f"[hand_node] Publicando ataque: {msg.data}")
        self.attack_pub.publish(msg)


def main():
    rospy.init_node("hand_node", anonymous=True)
    node = HandNode()
    rospy.loginfo("[hand_node] Nodo de mano iniciado.")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
