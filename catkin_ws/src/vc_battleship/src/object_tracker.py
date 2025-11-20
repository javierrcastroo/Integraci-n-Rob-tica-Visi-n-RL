# object_tracker.py
import cv2
import numpy as np

# color por defecto de objetos genéricos (seguimos permitiendo la tecla 'o')
OBJ_LOWER_DEFAULT = np.array([0, 120, 80], dtype=np.uint8)
OBJ_UPPER_DEFAULT = np.array([15, 255, 255], dtype=np.uint8)

# color por defecto del origen (luego lo calibras con 'r')
ORIG_LOWER_DEFAULT = np.array([90, 120, 80], dtype=np.uint8)   # azul por defecto
ORIG_UPPER_DEFAULT = np.array([130, 255, 255], dtype=np.uint8)

current_obj_lower = OBJ_LOWER_DEFAULT.copy()
current_obj_upper = OBJ_UPPER_DEFAULT.copy()

current_origin_lower = ORIG_LOWER_DEFAULT.copy()
current_origin_upper = ORIG_UPPER_DEFAULT.copy()

# rangos adicionales para los barcos detectados por color
SHIP_COLOR_RANGES = {
    "ship2": {"lower": None, "upper": None},
    "ship1": {"lower": None, "upper": None},
}

AMMO_COLOR_RANGE = {"lower": None, "upper": None}


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


def calibrate_object_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_origin_color_from_roi(hsv_roi):
    return _calibrate_from_roi(hsv_roi)


def calibrate_ship_color_from_roi(ship_type, hsv_roi):
    """Guarda el rango HSV asociado a un tipo de barco concreto."""
    if ship_type not in SHIP_COLOR_RANGES:
        raise ValueError(f"Tipo de barco desconocido: {ship_type}")
    lower, upper = _calibrate_from_roi(hsv_roi)
    SHIP_COLOR_RANGES[ship_type]["lower"] = lower
    SHIP_COLOR_RANGES[ship_type]["upper"] = upper
    return lower, upper


def get_ship_range(ship_type):
    data = SHIP_COLOR_RANGES.get(ship_type, {})
    return data.get("lower"), data.get("upper")


def calibrate_ammo_color_from_roi(hsv_roi):
    lower, upper = _calibrate_from_roi(hsv_roi)
    AMMO_COLOR_RANGE["lower"] = lower
    AMMO_COLOR_RANGE["upper"] = upper
    return lower, upper


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


def detect_ships_in_board(hsv_frame, board_quad, max_objs_per_type=4, min_area=50):
    """Detecta los barcos diferenciados por color calibrado."""
    detections = []
    combined_mask = None

    for ship_type in ("ship2", "ship1"):
        lower, upper = get_ship_range(ship_type)
        if lower is None or upper is None:
            continue

        centers, mask = detect_colored_points_in_board(
            hsv_frame,
            board_quad,
            lower,
            upper,
            max_objs=max_objs_per_type,
            min_area=min_area,
        )

        if mask is not None:
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

        for cx, cy in centers:
            detections.append({"pt": (cx, cy), "label": ship_type})

    # fallback al color genérico si no hay calibraciones específicas
    if not detections and current_obj_lower is not None and current_obj_upper is not None:
        centers, mask = detect_colored_points_in_board(
            hsv_frame,
            board_quad,
            current_obj_lower,
            current_obj_upper,
            max_objs=max_objs_per_type,
            min_area=min_area,
        )
        if mask is not None:
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)
        for cx, cy in centers:
            detections.append({"pt": (cx, cy), "label": "generic"})

    return detections, combined_mask


def detect_ammo_in_scene(hsv_frame, max_objs=6, min_area=80):
    lower = AMMO_COLOR_RANGE.get("lower")
    upper = AMMO_COLOR_RANGE.get("upper")
    if lower is None or upper is None:
        return [], None

    mask = cv2.inRange(hsv_frame, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = keep_largest_components(mask, k=max_objs, min_area=min_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    detections = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        detections.append({"pt": (cx, cy), "label": "ammo"})
        if len(detections) >= max_objs:
            break

    return detections, mask
