# object_tracker.py
import cv2
import numpy as np

# color por defecto del barco largo (lo calibras con '2')
SHIP_TWO_LOWER_DEFAULT = np.array([0, 120, 80], dtype=np.uint8)
SHIP_TWO_UPPER_DEFAULT = np.array([15, 255, 255], dtype=np.uint8)

# color por defecto de los barcos de una casilla (lo calibras con '1')
SHIP_ONE_LOWER_DEFAULT = np.array([100, 80, 80], dtype=np.uint8)
SHIP_ONE_UPPER_DEFAULT = np.array([125, 255, 255], dtype=np.uint8)

# color por defecto de la munición (lo calibras con 'm')
AMMO_LOWER_DEFAULT = np.array([40, 80, 60], dtype=np.uint8)
AMMO_UPPER_DEFAULT = np.array([80, 255, 255], dtype=np.uint8)

# color por defecto del origen (luego lo calibras con 'r')
ORIG_LOWER_DEFAULT = np.array([90, 120, 80], dtype=np.uint8)   # azul por defecto
ORIG_UPPER_DEFAULT = np.array([130, 255, 255], dtype=np.uint8)

current_ship_two_lower = SHIP_TWO_LOWER_DEFAULT.copy()
current_ship_two_upper = SHIP_TWO_UPPER_DEFAULT.copy()

current_ship_one_lower = SHIP_ONE_LOWER_DEFAULT.copy()
current_ship_one_upper = SHIP_ONE_UPPER_DEFAULT.copy()

current_origin_lower = ORIG_LOWER_DEFAULT.copy()
current_origin_upper = ORIG_UPPER_DEFAULT.copy()

current_ammo_lower = AMMO_LOWER_DEFAULT.copy()
current_ammo_upper = AMMO_UPPER_DEFAULT.copy()


def _calibrate_from_roi(hsv_roi, p_low=5, p_high=95, margin_h=3, margin_sv=20):
    H = hsv_roi[:,:,0].reshape(-1)
    S = hsv_roi[:,:,1].reshape(-1)
    V = hsv_roi[:,:,2].reshape(-1)

    mask_valid = (V > 40) & (S > 20)
    if mask_valid.sum() > 200:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]

    h_lo, h_hi = np.percentile(H, [p_low, p_high]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_low, p_high]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_low, p_high]).astype(int)

    h_lo = max(0, h_lo - margin_h)
    h_hi = min(179, h_hi + margin_h)
    s_lo = max(0, s_lo - margin_sv)
    s_hi = min(255, s_hi + margin_sv)
    v_lo = max(0, v_lo - margin_sv)
    v_hi = min(255, v_hi + margin_sv)

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return lower, upper


def calibrate_ship_two_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_ship_one_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_origin_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_ammo_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def detect_colored_points_global(hsv_frame, lower, upper, max_objs=8, min_area=40):
    """Detecta blobs del color dado en todo el frame HSV."""
    mask = cv2.inRange(hsv_frame, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = keep_largest_components(mask, k=max_objs, min_area=min_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        if len(centers) >= max_objs:
            break

    return centers, mask


def keep_largest_components(mask, k=4, min_area=50):
    """
    Devuelve una máscara nueva con solo las k componentes más grandes
    por área (en píxeles) que superen min_area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_mask = np.zeros_like(mask)
    kept = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        cv2.drawContours(new_mask, [c], -1, 255, -1)
        kept += 1
        if kept >= k:
            break
    return new_mask


def detect_colored_points_in_board(hsv_frame, board_quad, lower, upper,
                                   max_objs=4, min_area=50):
    """
    hsv_frame: frame del tablero en HSV
    board_quad: 4x2 float32 (tl,tr,br,bl)
    lower/upper: rango HSV del objeto/origen

    Devuelve:
      centers: lista de (x,y) en coordenadas de imagen
      mask: máscara de ese color (para debug)
    """
    # 1) máscara por color
    mask = cv2.inRange(hsv_frame, lower, upper)

    # 2) pequeña limpieza morfológica
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3,3), np.uint8), iterations=1)

    # 3) quedarnos solo con las k componentes más grandes
    mask = keep_largest_components(mask, k=max_objs, min_area=min_area)

    # 4) contornos finales
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    board_poly = np.array(board_quad, dtype=np.float32)

    # ordenar por área desc
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centers = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # comprobar si está dentro del tablero
        if cv2.pointPolygonTest(board_poly, (cx, cy), False) >= 0:
            centers.append((cx, cy))

        if len(centers) >= max_objs:
            break

    return centers, mask