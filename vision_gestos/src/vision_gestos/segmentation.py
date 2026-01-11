# segmentation.py
import cv2
import numpy as np

KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def calibrate_from_roi(hsv_roi, p_low=5, p_high=95, margin_h=5, margin_sv=20):
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    mask_valid = (V > 30) & (V < 245) & (S > 20)
    if mask_valid.sum() > 200:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]

    h_lo, h_hi = np.percentile(H, [p_low, p_high]).astype(int)
    s_lo, s_hi = np.percentile(S, [p_low, p_high]).astype(int)
    v_lo, v_hi = np.percentile(V, [p_low, p_high]).astype(int)

    h_lo = clamp(h_lo - margin_h, 0, 179)
    h_hi = clamp(h_hi + margin_h, 0, 179)
    s_lo = clamp(s_lo - margin_sv, 0, 255)
    s_hi = clamp(s_hi + margin_sv, 0, 255)
    v_lo = clamp(v_lo - margin_sv, 0, 255)
    v_hi = clamp(v_hi + margin_sv, 0, 255)

    if h_lo > h_hi:
        h_lo, h_hi = 0, max(h_lo, h_hi)

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return lower, upper


def hsv_medians(hsv_roi):
    """Devuelve la mediana de cada canal para un ROI en HSV.

    Filtra valores extremadamente oscuros o saturados para evitar que el papel
    quemado (muy brillante) o las sombras profundas distorsionen la estadística.
    """
    H = hsv_roi[:, :, 0].reshape(-1)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    mask_valid = (V > 20) & (V < 245) & (S < 245)
    if mask_valid.sum() > 50:
        H, S, V = H[mask_valid], S[mask_valid], V[mask_valid]

    return (
        int(np.median(H)),
        int(np.median(S)),
        int(np.median(V)),
    )


def apply_white_reference(lower_skin, upper_skin, ref_info, hsv_frame,
                          max_shift=(12, 30, 30)):
    """Ajusta dinámicamente el rango HSV de la piel usando un blanco de referencia.

    - `ref_info` debe tener las claves `median` (HSV base del papel) y `roi`
      (x0, x1, y0, y1) donde se encuentra el papel en la imagen.
    - `max_shift` limita cuánto se puede desplazar cada canal para evitar derivas
      agresivas por ruido o cambios puntuales.
    """
    if lower_skin is None or upper_skin is None or ref_info is None:
        return lower_skin, upper_skin, (0, 0, 0)

    x0, x1, y0, y1 = ref_info.get("roi", (0, 0, 0, 0))
    h, w = hsv_frame.shape[:2]
    x0, x1 = sorted((clamp(x0, 0, w), clamp(x1, 0, w)))
    y0, y1 = sorted((clamp(y0, 0, h), clamp(y1, 0, h)))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return lower_skin, upper_skin, (0, 0, 0)

    roi = hsv_frame[y0:y1, x0:x1]
    current_med = hsv_medians(roi)
    base_med = ref_info.get("median", (0, 0, 0))

    delta = [
        clamp(current_med[i] - base_med[i], -max_shift[i], max_shift[i])
        for i in range(3)
    ]

    new_lower = np.array([
        clamp(int(lower_skin[i]) + delta[i], 0, 179 if i == 0 else 255)
        for i in range(3)
    ], dtype=np.uint8)
    new_upper = np.array([
        clamp(int(upper_skin[i]) + delta[i], 0, 179 if i == 0 else 255)
        for i in range(3)
    ], dtype=np.uint8)

    return new_lower, new_upper, tuple(delta)

def largest_component_mask(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)

def segment_hand_mask(hsv_frame, lower_skin, upper_skin):
    if lower_skin is None or upper_skin is None:
        return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    raw_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(raw_mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    mask_largest = largest_component_mask(mask)
    return mask_largest