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

def draw_hud(img, lower_skin, upper_skin, current_label, sequence_armed, acciones_len):
    if sequence_armed:
        hud = (
            "ROI: arrastra | 'c' calib | 'g' guarda | 'a' add | 'p' json | 'q' salir"
            " | Auto armado"
        )
    else:
        hud = (
            "ROI: arrastra | 'c' calib | 'g' guarda | 'a' add | 'p' json | 'q' salir"
            " | Haz 'demonio' para armar"
        )
    cv2.putText(img, hud, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Label actual: {current_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Lista auto: {acciones_len}/2",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )

    if lower_skin is not None and upper_skin is not None:
        cv2.putText(img,
                    f"HSV low:{tuple(int(v) for v in lower_skin)} up:{tuple(int(v) for v in upper_skin)}",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 255, 180), 1, cv2.LINE_AA)

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

def append_action(acciones, stable_label):
    if stable_label is None:
        print("[WARN] No hay gesto estable para añadir.")
        return acciones
    acciones.append(stable_label)
    print(f"[INFO] Añadido gesto a la lista: {stable_label}")
    print(f"[INFO] Lista actual: {acciones}")
    return acciones
