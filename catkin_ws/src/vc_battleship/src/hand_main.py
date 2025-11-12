# hand_main.py
import cv2
import os
import numpy as np

from hand_config import (
    PREVIEW_W, PREVIEW_H,
    RECOGNIZE_MODE,
    CONFIDENCE_THRESHOLD,
    USE_UNDISTORT_HAND,
    HAND_CAMERA_PARAMS_PATH,
)

import ui
from segmentation import calibrate_from_roi, segment_hand_mask
from features import compute_feature_vector
from classifier import knn_predict
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
from collections import deque


def majority_vote(labels):
    if not labels:
        return None
    return max(set(labels), key=labels.count)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 0 (mano)")

    HAND_CAM_MTX = HAND_DIST = None
    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        data = np.load(HAND_CAMERA_PARAMS_PATH)
        HAND_CAM_MTX = data["camera_matrix"]
        HAND_DIST = data["dist_coeffs"]
        print("[INFO] Undistort activado para la mano")

    # estado
    lower_skin = upper_skin = None
    gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
    current_label = "2dedos"
    acciones = []
    recent_preds = deque(maxlen=7)
    stable_history = deque(maxlen=150)
    sequence_armed = False
    last_auto_saved_label = None

    ARM_GESTURE = "demonio"
    SAVE_GESTURE = "cool"

    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", ui.mouse_callback)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # undistort
        if HAND_CAM_MTX is not None:
            frame = cv2.undistort(frame, HAND_CAM_MTX, HAND_DIST)

        # espejo + resize
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ROI
        ui.draw_roi_rectangle(vis)

        # segmentar mano
        mask = segment_hand_mask(hsv, lower_skin, upper_skin)
        ui.draw_hand_box(vis, mask)
        skin_only = cv2.bitwise_and(frame, frame, mask=mask)

        # features
        feat_vec = compute_feature_vector(mask)

        # reconocimiento
        best_dist = None
        per_frame_label = None
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

        if per_frame_label is not None:
            recent_preds.append(per_frame_label)
        stable_label = majority_vote(list(recent_preds))
        stable_history.append(stable_label)

        if (
            stable_label not in (None, "????")
            and last_auto_saved_label is not None
            and stable_label != last_auto_saved_label
        ):
            last_auto_saved_label = None

        if stable_label == ARM_GESTURE and not sequence_armed:
            sequence_armed = True
            acciones.clear()
            last_auto_saved_label = None
            stable_history.clear()
            print("[INFO] Secuencia armada tras gesto 'demonio'.")

        if stable_label == SAVE_GESTURE and sequence_armed:
            if acciones:
                save_sequence_json(acciones)
                print("[INFO] Secuencia guardada tras gesto 'cool':", acciones)
            acciones.clear()
            sequence_armed = False
            stable_history.clear()
            last_auto_saved_label = None
            print("[INFO] Secuencia reiniciada, realiza 'demonio' para armar de nuevo.")

        if sequence_armed:
            if (
                stable_history
                and len(stable_history) == stable_history.maxlen
                and stable_history[0] not in (None, "????", ARM_GESTURE, SAVE_GESTURE)
                and all(lbl == stable_history[0] for lbl in stable_history)
            ):
                label_to_add = stable_history[0]
                if label_to_add == last_auto_saved_label:
                    stable_history.clear()
                elif len(acciones) < 2:
                    acciones = ui.append_action(acciones, label_to_add)
                    last_auto_saved_label = label_to_add
                    stable_history.clear()
                else:
                    print("[WARN] Lista llena. Usa gesto 'cool' para guardar y reiniciar.")

        # HUD
        ui.draw_hud(
            vis,
            lower_skin,
            upper_skin,
            current_label,
            sequence_armed,
            len(acciones),
        )
        ui.draw_prediction(vis, stable_label, best_dist if best_dist else 0.0)

        # mostrar
        cv2.imshow("Mano", vis)
        cv2.imshow("Mascara mano", mask)
        cv2.imshow("Solo piel mano", skin_only)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # -------- teclas de mano --------
        if key == ord('c'):
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    lower_skin, upper_skin = calibrate_from_roi(roi_hsv)
                    print("[INFO] calibrado HSV mano:", lower_skin, upper_skin)
                else:
                    print("[WARN] ROI muy pequeño")
            else:
                print("[WARN] dibuja un ROI en 'Mano' primero")

        elif key == ord('g'):
            if feat_vec is not None:
                save_gesture_example(feat_vec, current_label)
                if RECOGNIZE_MODE:
                    gallery.append((feat_vec, current_label))
                print(f"[INFO] guardado gesto {current_label}")
            else:
                print("[WARN] no hay gesto válido")

        elif key == ord('a'):
            if not sequence_armed:
                print("[WARN] Necesitas hacer el gesto 'demonio' antes de añadir gestos.")
            elif len(acciones) >= 2:
                print("[WARN] Lista llena. Usa gesto 'cool' para guardar y reiniciar.")
            else:
                acciones = ui.append_action(acciones, stable_label)
                if stable_label not in (None, "????"):
                    last_auto_saved_label = stable_label

        elif key == ord('p'):
            if not acciones:
                print("[WARN] No hay acciones para guardar.")
            else:
                save_sequence_json(acciones)
                print("[INFO] secuencia guardada manualmente:", acciones)
                acciones.clear()
                sequence_armed = False
                stable_history.clear()
                last_auto_saved_label = None
                print("[INFO] Secuencia reiniciada, realiza 'demonio' para armar de nuevo.")

        elif key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('d'), ord('p'), ord('-')):
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
            }
            current_label = mapping[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
