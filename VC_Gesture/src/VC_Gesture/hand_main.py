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
from segmentation import (
    calibrate_from_roi,
    segment_hand_mask,
)
from features import compute_feature_vector
from classifier import knn_predict
from storage import save_gesture_example, load_gesture_gallery, save_sequence_json
from collections import deque

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
    white_ref = None
    gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
    current_label = "2dedos"
    acciones = []
    recent_preds = deque(maxlen=7)
    capture_state = "STANDBY"
    pending_candidate = None
    gesture_window = GestureWindow()
    status_lines = ["Standby: haz 'demond' para activar el registro."]

    def set_state(new_state, lines):
        nonlocal capture_state, status_lines
        capture_state = new_state
        status_lines = lines
        gesture_window.reset()

    def set_status(lines):
        nonlocal status_lines
        status_lines = lines

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

        # segmentar mano con el HSV calibrado
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

        # HUD
        ui.draw_hud(
            vis,
            lower_skin,
            upper_skin,
            current_label,
        )
        ui.draw_prediction(vis, stable_label, best_dist if best_dist else 0.0)
        ui.draw_sequence_status(
            vis,
            acciones,
            capture_state,
            pending_candidate,
            status_lines,
            gesture_window.progress(),
        )

        # mostrar
        cv2.imshow("Mano", vis)
        cv2.imshow("Mascara mano", mask)
        cv2.imshow("Solo piel mano", skin_only)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # -------- flujo controlado por gestos --------
        resolved_label = gesture_window.push(stable_label)

        if resolved_label is not None:
            if capture_state == "STANDBY":
                if resolved_label in TRIGGER_GESTURES:
                    set_state("CAPTURA", ["Sistema activo: muestra el primer gesto."])
                else:
                    set_status(["Sigue en standby, haz 'demond' para comenzar."])

            elif capture_state == "CAPTURA":
                if resolved_label == "????" or resolved_label in CONTROL_GESTURES:
                    set_status(["Gesto no válido, repítelo."])
                else:
                    pending_candidate = resolved_label
                    set_state(
                        "CONFIRMACION",
                        [
                            f"¿Tu gesto es '{pending_candidate}'?",
                            "Confirma con 'ok' o repite con 'nook'.",
                        ],
                    )

            elif capture_state == "CONFIRMACION":
                if resolved_label == CONFIRM_GESTURE and pending_candidate:
                    acciones.append(pending_candidate)
                    print(f"[INFO] Añadido gesto confirmado: {pending_candidate}")
                    pending_candidate = None
                    if len(acciones) >= MAX_SEQUENCE_LENGTH:
                        set_state(
                            "COOL",
                            ["Secuencia completa, haz 'cool' para imprimirla."],
                        )
                    else:
                        set_state("CAPTURA", ["Gesto guardado. Muestra el siguiente gesto."])
                elif resolved_label == REJECT_GESTURE:
                    print("[INFO] Gesto rechazado, repite el anterior.")
                    pending_candidate = None
                    set_state("CAPTURA", ["Repite el gesto a registrar."])
                else:
                    set_status(["Se esperaba 'ok' o 'nook'."])

            elif capture_state == "COOL":
                if resolved_label == PRINT_GESTURE and len(acciones) == MAX_SEQUENCE_LENGTH:
                    print("[INFO] Secuencia final:", acciones)
                    save_sequence_json(acciones)
                    acciones.clear()
                    pending_candidate = None
                    set_state(
                        "STANDBY",
                        ["Standby: haz 'demond' para activar un nuevo registro."],
                    )
                else:
                    set_status(["Secuencia lista. Usa 'cool' para imprimirla."])

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

        elif key == ord('b'):
            if ui.roi_defined:
                x0, x1 = sorted([ui.x_start, ui.x_end])
                y0, y1 = sorted([ui.y_start, ui.y_end])
                if (x1 - x0) > 5 and (y1 - y0) > 5:
                    roi_hsv = hsv[y0:y1, x0:x1]
                    white_ref = {
                        "median": hsv_medians(roi_hsv),
                        "roi": (x0, x1, y0, y1),
                    }
                    print("[INFO] calibrado blanco de referencia en ROI:", white_ref["median"])
                else:
                    print("[WARN] ROI muy pequeño para referencia blanca")
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
            current_label = mapping[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()