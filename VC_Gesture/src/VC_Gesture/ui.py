# ui.py
import cv2

# estado global del ROI
roi_selecting = False
roi_defined = False
x_start = y_start = x_end = y_end = 0

def mouse_callback(event, x, y, flags, param):
    global roi_selecting, roi_defined, x_start, y_start, x_end, y_end
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selecting = True
        roi_defined = False
        x_start, y_start = x, y
        x_end, y_end = x, y
    elif event == cv2.EVENT_MOUSEMOVE and roi_selecting:
        x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selecting = False
        roi_defined = True
        x_end, y_end = x, y

def draw_roi_rectangle(img):
    if roi_selecting or roi_defined:
        cv2.rectangle(img,
                      (x_start, y_start),
                      (x_end, y_end),
                      (0, 255, 255), 2)

def draw_hud(img, lower_skin, upper_skin, current_label):
    hud = "ROI: arrastra | 'c' calib mano | 'g' guarda muestra | 'q' salir"
    cv2.putText(img, hud, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Label actual: {current_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 255, 180), 1, cv2.LINE_AA)

    if lower_skin is not None and upper_skin is not None:
        base_txt = f"HSV low:{tuple(int(v) for v in lower_skin)} up:{tuple(int(v) for v in upper_skin)}"
        cv2.putText(img,
                    base_txt,
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 200, 255), 1, cv2.LINE_AA)

    if white_ref_ready:
        cv2.putText(img,
                    f"Ref blanco activa. Delta HSV: {shift_delta}",
                    (10, img.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 220, 180), 1, cv2.LINE_AA)

def draw_prediction(img, label, dist):
    if label is None:
        return
    txt = f"Gesto: {label} (dist={dist:.2f})"
    cv2.putText(img, txt, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 1, cv2.LINE_AA)

def draw_hand_box(img, mask):
    # dibuja rect mínimo alrededor de la mano (para debug)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.polylines(img, [box], True, (255, 0, 255), 2)

def draw_sequence_status(img, acciones, capture_state, pending, status_lines, progress):
    y = 100
    seq_text = ", ".join(acciones) if acciones else "(vacía)"
    cv2.putText(img,
                f"Secuencia (max 2): {seq_text}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA)
    y += 20
    cv2.putText(img,
                f"Estado: {capture_state}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
                cv2.LINE_AA)
    if pending:
        y += 20
        cv2.putText(img,
                    f"Pendiente: {pending}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 255, 200),
                    1,
                    cv2.LINE_AA)
    for line in status_lines[:2]:
        if not line:
            continue
        y += 20
        cv2.putText(img,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA)

    bar_x = 10
    bar_y = y + 30
    bar_w = 200
    bar_h = 10
    cv2.rectangle(img,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (100, 100, 100),
                  1)
    fill_w = int(bar_w * max(0.0, min(1.0, progress)))
    cv2.rectangle(img,
                  (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h),
                  (0, 200, 0),
                  -1)
    cv2.putText(img,
                f"Ventana 150f: {int(progress * 100)}%",
                (bar_x + 5, bar_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                1,
                cv2.LINE_AA)