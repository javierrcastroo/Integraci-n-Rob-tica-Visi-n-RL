# hand_main.py
import cv2
import os
import numpy as np

from hand_config import (
    PREVIEW_W,
    PREVIEW_H,
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
from hand_pipeline import GestureConsensus, GestureSequenceManager


def handle_sequence_events(events, consensus):
    for event in events:
        if event.kind == "save" and event.actions:
            save_sequence_json(event.actions)
        if event.message:
            print(event.message)
        if event.reset_consensus:
            consensus.clear()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 0 (mano)")

    hand_cam_mtx = hand_dist = None
    undistort_map1 = undistort_map2 = None
    if USE_UNDISTORT_HAND and os.path.exists(HAND_CAMERA_PARAMS_PATH):
        data = np.load(HAND_CAMERA_PARAMS_PATH)
        hand_cam_mtx = data["camera_matrix"]
        hand_dist = data["dist_coeffs"]
        print("[INFO] Undistort activado para la mano")

    lower_skin = upper_skin = None
    gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
    current_label = "2dedos"

    consensus = GestureConsensus()
    sequence_manager = GestureSequenceManager()

    cv2.namedWindow("Mano")
    cv2.setMouseCallback("Mano", ui.mouse_callback)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if hand_cam_mtx is not None:
            if undistort_map1 is None:
                h, w = frame.shape[:2]
                undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
                    hand_cam_mtx,
                    hand_dist,
                    None,
                    hand_cam_mtx,
                    (w, h),
                    cv2.CV_16SC2,
                )
            frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ui.draw_roi_rectangle(vis)

        mask = segment_hand_mask(hsv, lower_skin, upper_skin)
        ui.draw_hand_box(vis, mask)
        skin_only = cv2.bitwise_and(frame, frame, mask=mask)

        feat_vec = compute_feature_vector(mask)

        best_dist = None
        per_frame_label = None
        if feat_vec is not None and RECOGNIZE_MODE:
            raw_label, best_dist = knn_predict(feat_vec, gallery, k=5)
            if raw_label is not None and best_dist is not None:
                per_frame_label = raw_label if best_dist <= CONFIDENCE_THRESHOLD else "????"

        stable_label, candidate_label, count = consensus.step(per_frame_label)
        if candidate_label is not None:
            events = sequence_manager.handle_candidate(candidate_label, count)
            handle_sequence_events(events, consensus)

        ui.draw_hud(
            vis,
            lower_skin,
            upper_skin,
            current_label,
            sequence_manager.armed,
            sequence_manager.action_count,
            sequence_manager.pending_label,
        )
        ui.draw_prediction(vis, stable_label, best_dist if best_dist else 0.0)

        cv2.imshow("Mano", vis)
        cv2.imshow("Mascara mano", mask)
        cv2.imshow("Solo piel mano", skin_only)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("c"):
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

        elif key == ord("g"):
            if feat_vec is not None:
                save_gesture_example(feat_vec, current_label)
                if RECOGNIZE_MODE:
                    gallery.append((feat_vec, current_label))
                print(f"[INFO] guardado gesto {current_label}")
            else:
                print("[WARN] no hay gesto válido")

        elif key == ord("a"):
            label_to_add = (
                sequence_manager.pending_label
                if sequence_manager.pending_label not in (None, "????")
                else stable_label
            )
            events = sequence_manager.manual_add(label_to_add)
            handle_sequence_events(events, consensus)

        elif key == ord("p"):
            events = sequence_manager.manual_save()
            handle_sequence_events(events, consensus)

        elif key in (
            ord("0"),
            ord("1"),
            ord("2"),
            ord("3"),
            ord("4"),
            ord("5"),
            ord("d"),
            ord("p"),
            ord("-"),
        ):
            mapping = {
                ord("0"): "0dedos",
                ord("1"): "1dedo",
                ord("2"): "2dedos",
                ord("3"): "3dedos",
                ord("4"): "4dedos",
                ord("5"): "5dedos",
                ord("d"): "demonio",
                ord("p"): "ok",
                ord("n"): "nook",
                ord("-"): "cool",
            }
            current_label = mapping.get(key, current_label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
