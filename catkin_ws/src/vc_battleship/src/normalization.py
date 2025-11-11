# normalization.py
import cv2
import numpy as np

def extract_normalized_hand(mask, size=200):
    """
    Recibe una máscara 0/255 y devuelve:
      - mask_rot_resized: máscara rotada y reescalada a (size, size)
      - contour_rot: contorno principal ya en esa máscara rotada
    Estrategia:
      1. buscamos el contorno más grande en la máscara original
      2. calculamos el rectángulo mínimo y su ángulo
      3. rotamos **la máscara completa**
      4. en la máscara rotada volvemos a buscar el contorno
      5. recortamos y reescalamos
    Así evitamos hacer cv2.transform sobre el contorno (que es lo que te ha petado).
    """
    # 1) contorno más grande en la máscara original
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, None

    # 2) rectángulo mínimo para saber el ángulo
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    # corrección típica de OpenCV
    if angle < -45:
        angle += 90

    h, w = mask.shape[:2]
    center = (w // 2, h // 2)

    # 3) rotar la máscara entera
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # 4) volver a buscar contorno en la máscara rotada
    contours_rot, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rot:
        return None, None
    c_rot = max(contours_rot, key=cv2.contourArea)
    if cv2.contourArea(c_rot) < 100:
        return None, None

    # 5) recortar el bounding rect simple (no el minAreaRect) en la máscara rotada
    x, y, rw, rh = cv2.boundingRect(c_rot)
    hand_roi = rotated[y:y+rh, x:x+rw]
    if hand_roi.size == 0:
        return None, None

    # 6) reescalar a tamaño fijo
    resized = cv2.resize(hand_roi, (size, size), interpolation=cv2.INTER_NEAREST)

    return resized, c_rot