# board_ui.py
import cv2

# ====== ROI PARA COLOR DEL TABLERO (clic izquierdo) ======
board_roi_selecting = False
board_roi_defined = False
bx_start = by_start = bx_end = by_end = 0

# ====== PUNTOS DE MEDIDA (clic derecho) ======
measure_points = []   # lista de (x, y)


def board_mouse_callback(event, x, y, flags, param):
    """
    - Botón izquierdo: definir ROI del tablero (para calibrar HSV con 'b', 'o', 'm', ...)
    - Botón derecho: añadir punto de medida
    """
    global board_roi_selecting, board_roi_defined
    global bx_start, by_start, bx_end, by_end
    global measure_points

    if event == cv2.EVENT_LBUTTONDOWN:
        board_roi_selecting = True
        board_roi_defined = False
        bx_start, by_start = x, y
        bx_end, by_end = x, y

    elif event == cv2.EVENT_MOUSEMOVE and board_roi_selecting:
        bx_end, by_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        board_roi_selecting = False
        board_roi_defined = True
        bx_end, by_end = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        # añadir punto de medida
        measure_points.append((x, y))
        # si hay más de 2, reseteamos para empezar otra medida
        if len(measure_points) > 2:
            measure_points = [(x, y)]  # empezamos de nuevo solo con este


def draw_board_roi(img):
    """
    Dibuja el rectángulo del ROI si se está seleccionando.
    """
    if board_roi_selecting or board_roi_defined:
        cv2.rectangle(
            img,
            (bx_start, by_start),
            (bx_end,   by_end),
            (0, 255, 255),
            2
        )


def draw_measure_points(img):
    """
    Dibuja los puntos de medida (clic derecho) sobre la imagen del tablero.
    """
    for i, (x, y) in enumerate(measure_points):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"P{i+1}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


def draw_board_hud(img):
    """
    HUD con las teclas específicas del tablero.
    """
    y0 = 20
    dy = 18
    cv2.putText(img, "TABLERO:", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "b: calibrar color tablero (ROI izq)", (10, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "r: calibrar ORIGEN (ROI marcador)", (10, y0 + 2*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "2: calibrar BARCO x2 (ROI barco)", (10, y0 + 3*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "1: calibrar BARCO x1 (ROI barco)", (10, y0 + 4*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "m: calibrar MUN (ROI municion)", (10, y0 + 5*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img, "s: enviar layout (150 frames)", (10, y0 + 6*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 1, cv2.LINE_AA)
    cv2.putText(img, "Click izq: definir ROI", (10, y0 + 7*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)
    cv2.putText(img, "Click dcho: punto de medida", (10, y0 + 8*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)


def draw_capture_status(img, state, progress, lines):
    y_base = 180
    cv2.putText(
        img,
        f"Estado tablero: {state}",
        (10, y_base),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    bar_x0, bar_y0 = 10, y_base + 15
    bar_w, bar_h = 260, 14
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 80), 1)
    fill_w = int(bar_w * max(0.0, min(1.0, progress)))
    cv2.rectangle(
        img,
        (bar_x0, bar_y0),
        (bar_x0 + fill_w, bar_y0 + bar_h),
        (0, 200, 0),
        -1,
    )

    for idx, line in enumerate(lines[:3]):
        cv2.putText(
            img,
            line,
            (10, y_base + 40 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_validation_result(img, quad, text, is_valid):
    if quad is None:
        return
    color = (0, 255, 0) if is_valid else (0, 0, 255)
    tl = quad[0]
    x = int(tl[0])
    y = int(tl[1]) - 10
    cv2.putText(
        img,
        text,
        (x, max(20, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )