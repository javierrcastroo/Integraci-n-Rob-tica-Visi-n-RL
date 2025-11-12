#!/usr/bin/env python
import os
from collections import Counter, deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from classifier import knn_predict
from features import compute_feature_vector
from hand_config import (
    PREVIEW_W,
    PREVIEW_H,
    RECOGNIZE_MODE,
    CONFIDENCE_THRESHOLD,
    USE_UNDISTORT_HAND,
    HAND_CAMERA_PARAMS_PATH,
)
from segmentation import calibrate_from_roi, segment_hand_mask
import ui


def majority_vote(labels):
    if not labels:
        return None
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def majority_label_with_exclusions(labels, extra_invalid=()):
    invalid = {None, "????"}
    invalid.update(extra_invalid)
    filtered = [lbl for lbl in labels if lbl not in invalid]
    if not filtered:
        return None, 0
    counts = Counter(filtered)
    return counts.most_common(1)[0]


class GestureNode:
    ARM_GESTURE = "demonio"
    SAVE_GESTURE = "cool"

    def __init__(self):
        rospy.init_node("gesture_node", anonymous=True)

        self.bridge = CvBridge()
        self.last_frame_bgr = None

        self.pub_sequence = rospy.Publisher("/gesture/attack_list", String, queue_size=10)
        rospy.Subscriber("/webcam/image_raw", Image, self.cb_image, queue_size=1)

        self.gallery = []
        if RECOGNIZE_MODE:
            from storage import load_gesture_gallery  # import tardío para evitar dependencias ROS

            self.gallery = load_gesture_gallery()
            rospy.loginfo("[GESTURE] Galería cargada con %d ejemplos.", len(self.gallery))

        self.lower_skin = None
        self.upper_skin = None

        self.recent_preds = deque(maxlen=7)
        self.stable_history = deque(maxlen=150)
        self.sequence_armed = False
        self.acciones = []
        self.last_auto_saved_label = None

        self.current_label = "2dedos"

        self.hand_cam_mtx = None
        self.hand_dist = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
            params = np.load(HAND_CAMERA_PARAMS_PATH)
            self.hand_cam_mtx = params["camera_matrix"]
            self.hand_dist = params["dist_coeffs"]
            rospy.loginfo("[GESTURE] Corrección de distorsión activada para la mano.")

        cv2.namedWindow("Gesture")
        cv2.namedWindow("Gesture_mask")
        cv2.namedWindow("Gesture_skin")
        cv2.setMouseCallback("Gesture", ui.mouse_callback)

    def cb_image(self, msg):
        self.last_frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def process_current_frame(self):
        frame = self.last_frame_bgr
        if frame is None:
            return None

        if self.hand_cam_mtx is not None:
            if self.undistort_map1 is None:
                h, w = frame.shape[:2]
                self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                    self.hand_cam_mtx,
                    self.hand_dist,
                    None,
                    self.hand_cam_mtx,
                    (w, h),
                    cv2.CV_16SC2,
                )
            frame = cv2.remap(frame, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR)

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))

        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ui.draw_roi_rectangle(vis)

        mask = segment_hand_mask(hsv, self.lower_skin, self.upper_skin)
        ui.draw_hand_box(vis, mask)
        skin_only = cv2.bitwise_and(frame, frame, mask=mask)

        feat_vec = compute_feature_vector(mask)

        best_dist = None
        per_frame_label = None
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, self.gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

        if per_frame_label is not None:
            self.recent_preds.append(per_frame_label)

        stable_label = majority_vote(list(self.recent_preds))
        self.stable_history.append(stable_label)

        return vis, mask, skin_only, hsv, stable_label, best_dist

    def handle_state_machine(self, stable_label):
        if (
            stable_label not in (None, "????")
            and self.last_auto_saved_label is not None
            and stable_label != self.last_auto_saved_label
        ):
            self.last_auto_saved_label = None

        if len(self.stable_history) == self.stable_history.maxlen:
            candidate_label, count = majority_label_with_exclusions(self.stable_history)

            if candidate_label is None:
                self.stable_history.clear()
                return

            if candidate_label == self.ARM_GESTURE:
                if not self.sequence_armed:
                    self.sequence_armed = True
                    self.acciones.clear()
                    self.last_auto_saved_label = None
                    rospy.loginfo(
                        "[GESTURE] Secuencia armada tras gesto '%s'.",
                        self.ARM_GESTURE,
                    )
                self.stable_history.clear()
                return

            if candidate_label == self.SAVE_GESTURE:
                if self.sequence_armed:
                    if self.acciones:
                        payload = String()
                        payload.data = str(self.acciones)
                        self.pub_sequence.publish(payload)
                        rospy.loginfo(
                            "[GESTURE] Secuencia enviada tras gesto '%s': %s",
                            self.SAVE_GESTURE,
                            self.acciones,
                        )
                    else:
                        rospy.logwarn(
                            "[GESTURE] Gesto '%s' pero la lista está vacía.",
                            self.SAVE_GESTURE,
                        )
                    self.acciones.clear()
                    self.sequence_armed = False
                    self.last_auto_saved_label = None
                    rospy.loginfo(
                        "[GESTURE] Secuencia reiniciada, realiza '%s' para armar de nuevo.",
                        self.ARM_GESTURE,
                    )
                else:
                    rospy.logwarn("[GESTURE] Ignorando '%s' sin armar la secuencia.", self.SAVE_GESTURE)
                self.stable_history.clear()
                return

            if not self.sequence_armed:
                rospy.logwarn(
                    "[GESTURE] Ignorando gesto '%s' sin armar la secuencia con '%s'.",
                    candidate_label,
                    self.ARM_GESTURE,
                )
                self.stable_history.clear()
                return

            if candidate_label == self.last_auto_saved_label:
                self.stable_history.clear()
                return

            if len(self.acciones) >= 2:
                rospy.logwarn(
                    "[GESTURE] Lista llena (%s). Usa '%s' para publicar y reiniciar.",
                    self.acciones,
                    self.SAVE_GESTURE,
                )
                self.stable_history.clear()
                return

            label_to_add, count = majority_label_with_exclusions(
                self.stable_history, (self.ARM_GESTURE, self.SAVE_GESTURE)
            )
            if label_to_add is None:
                self.stable_history.clear()
                return

            self.acciones = ui.append_action(self.acciones, label_to_add)
            self.last_auto_saved_label = label_to_add
            rospy.loginfo(
                "[GESTURE] Gesto mayoritario en %d frames: %s (cuenta=%d).",
                self.stable_history.maxlen,
                label_to_add,
                count,
            )
            self.stable_history.clear()

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            result = self.process_current_frame()
            if result is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("Salida solicitada")
                rate.sleep()
                continue

            vis, mask, skin_only, hsv, stable_label, best_dist = result

            self.handle_state_machine(stable_label)

            ui.draw_hud(
                vis,
                self.lower_skin,
                self.upper_skin,
                self.current_label,
                self.sequence_armed,
                len(self.acciones),
            )
            ui.draw_prediction(vis, stable_label, best_dist if best_dist else 0.0)

            cv2.imshow("Gesture", vis)
            cv2.imshow("Gesture_mask", mask)
            cv2.imshow("Gesture_skin", skin_only)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("Salida solicitada")
                break

            if key == ord("c"):
                if ui.roi_defined:
                    x0, x1 = sorted([ui.x_start, ui.x_end])
                    y0, y1 = sorted([ui.y_start, ui.y_end])
                    if (x1 - x0) > 5 and (y1 - y0) > 5:
                        roi_hsv = hsv[y0:y1, x0:x1]
                        self.lower_skin, self.upper_skin = calibrate_from_roi(roi_hsv)
                        rospy.loginfo(
                            "[GESTURE] HSV calibrado: %s %s",
                            self.lower_skin,
                            self.upper_skin,
                        )
                    else:
                        rospy.logwarn("[GESTURE] ROI muy pequeño para calibrar.")
                else:
                    rospy.logwarn("[GESTURE] Dibuja un ROI en 'Gesture' antes de calibrar.")

            rate.sleep()

        cv2.destroyAllWindows()


def main():
    node = GestureNode()
    node.run()


if __name__ == "__main__":
    main()

