#!/usr/bin/env python3
import cv2
import numpy as np

import board_state

"""
Utilidades para trabajar con marcadores ArUco y actualizar el ORIGEN
global del tablero (board_state.GLOBAL_ORIGIN) a partir de un ID concreto.
"""

# --------------------------------------------------------------------
# Inicialización compatible con varias versiones de OpenCV
# --------------------------------------------------------------------

# Diccionario ArUco
try:
    # API nueva (OpenCV >= 4.7)
    _ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
except AttributeError:
    # API clásica (la que suele venir con ROS Noetic / Ubuntu 20.04)
    _ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

# Parámetros del detector + posible ArucoDetector (API nueva)
_USE_NEW_API = False
_ARUCO_DETECTOR = None

try:
    # OpenCV reciente: cv2.aruco.DetectorParameters() existe
    _ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
    _USE_NEW_API = True
except AttributeError:
    # OpenCV antiguo: sólo existe DetectorParameters_create()
    _ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    _ARUCO_DETECTOR = None
    _USE_NEW_API = False


# --------------------------------------------------------------------
# Funciones internas de detección
# --------------------------------------------------------------------

def _detect_markers(gray):
    """
    Envuelve la llamada a detectMarkers para soportar tanto la API nueva
    (ArucoDetector) como la clásica (cv2.aruco.detectMarkers).
    Devuelve (corners, ids, rejected).
    """
    if _USE_NEW_API and _ARUCO_DETECTOR is not None:
        corners, ids, rejected = _ARUCO_DETECTOR.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            _ARUCO_DICT,
            parameters=_ARUCO_PARAMS
        )
    return corners, ids, rejected


def _marker_center(corners):
    """
    corners: array de shape (1, 4, 2) típico de detectMarkers.
    Devuelve (cx, cy) en píxeles.
    """
    pts = corners.reshape(-1, 2)  # (4, 2)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    return cx, cy


# --------------------------------------------------------------------
# API principal
# --------------------------------------------------------------------

def update_global_origin_from_aruco(frame_bgr, aruco_id=2, draw=False):
    """
    Busca un marcador ArUco con el ID dado en el frame BGR y, si lo encuentra,
    actualiza board_state.GLOBAL_ORIGIN con el centro del marcador en píxeles.

    :param frame_bgr: imagen en BGR (OpenCV) de la cámara del tablero
    :param aruco_id:  ID del marcador que define el ORIGEN (por defecto 2)
    :param draw:      si True, dibuja el marcador y el centro sobre la imagen
                      (útil para debug si luego usas esta imagen)
    :return: True si ha encontrado ese marcador y ha actualizado el origen,
             False en caso contrario.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _detect_markers(gray)

    if ids is None or len(ids) == 0:
        # No se ve ningún marcador
        return False

    found = False
    for corners, mid in zip(corners_list, ids.flatten()):
        if int(mid) == int(aruco_id):
            cx, cy = _marker_center(corners)
            board_state.GLOBAL_ORIGIN = (cx, cy)
            found = True

            if draw:
                # Dibuja el contorno y el centro para depuración
                cv2.polylines(
                    frame_bgr,
                    [corners.astype(np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.circle(frame_bgr, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(
                    frame_bgr,
                    f"ArUco {aruco_id}",
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            break

    return found
