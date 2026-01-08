# board_config.py
import os

# usar calibración de la cámara del tablero
USE_UNDISTORT_BOARD = False
BOARD_CAMERA_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params/camera_params.npz")

# tamaño de la vista aplanada opcional
WARP_SIZE = 500