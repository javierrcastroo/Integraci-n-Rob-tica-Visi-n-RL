# features.py
import cv2
import numpy as np
from normalization import extract_normalized_hand

def compute_hu_moments(contour):
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()
    return hu

def compute_radial_signature(mask, contour, num_angles=36):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return np.zeros(num_angles, dtype=np.float32)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    h, w = mask.shape[:2]
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    distances = []
    for theta in angles:
        dx = np.cos(theta)
        dy = np.sin(theta)
        dist = 0.0
        while True:
            x = int(round(cx + dist*dx))
            y = int(round(cy + dist*dy))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask[y, x] == 0:
                break
            dist += 1.0
        distances.append(dist)

    distances = np.array(distances, dtype=np.float32)
    m = distances.max() if distances.max() > 0 else 1.0
    return distances / m

def compute_feature_vector(mask):
    norm_mask, contour = extract_normalized_hand(mask)
    if norm_mask is None or contour is None:
        return None

    # encontrar contorno en la máscara normalizada
    contours, _ = cv2.findContours(norm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None

    hu = compute_hu_moments(c)
    radial = compute_radial_signature(norm_mask, c, 36)

    feat = np.concatenate([hu.astype(np.float64), radial.astype(np.float64)], axis=0)
    return feat  # ~43 dims