# aruco_utils.py
import cv2
import numpy as np
import board_state

# Diccionario de ArUco que vamos a usar 
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)


def _create_detector_params():
    """Devuelve un objeto DetectorParameters compatible con distintas versiones de OpenCV."""

    if hasattr(cv2.aruco, "DetectorParameters"):
        try:
            return cv2.aruco.DetectorParameters()
        except TypeError:
            # Algunas versiones exigen parámetros opcionales; probamos sin argumentos.
            pass
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise AttributeError("DetectorParameters no disponible en cv2.aruco")


_ARUCO_PARAMS = _create_detector_params()

# ID del marcador que usaremos como origen global
ARUCO_ORIGIN_ID = 1 


def detect_aruco_origin(frame, aruco_id=ARUCO_ORIGIN_ID, draw=True):
    """
    Detecta en 'frame' un marcador ArUco con ID = aruco_id.
    Si lo encuentra:
      - devuelve True, (cx, cy)
      - opcionalmente lo dibuja
    Si no:
      - devuelve False, None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)

    if ids is None or len(ids) == 0:
        return False, None

    ids = ids.flatten()
    for corner, marker_id in zip(corners, ids):
        if marker_id == aruco_id:
            pts = corner.reshape((4, 2))
            (tl, tr, br, bl) = pts
            # centro del marcador
            cx = int((tl[0] + br[0]) / 2.0)
            cy = int((tl[1] + br[1]) / 2.0)

            if draw:
                cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"ArUco {marker_id}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            return True, (cx, cy)

    return False, None


def update_global_origin_from_aruco(frame, aruco_id=ARUCO_ORIGIN_ID):
    """
    Llama a detect_aruco_origin y, si lo encuentra, actualiza board_state.GLOBAL_ORIGIN.
    Si no lo ve un frame, aguanta unos cuantos antes de borrarlo.
    """
    ok, pt = detect_aruco_origin(frame, aruco_id=aruco_id, draw=True)
    if ok:
        board_state.GLOBAL_ORIGIN = pt
        board_state.GLOBAL_ORIGIN_MISS = 0
    else:
        board_state.GLOBAL_ORIGIN_MISS += 1
        if board_state.GLOBAL_ORIGIN_MISS > board_state.GLOBAL_ORIGIN_MAX_MISS:
            board_state.GLOBAL_ORIGIN = None
