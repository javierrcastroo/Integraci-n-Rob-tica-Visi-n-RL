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