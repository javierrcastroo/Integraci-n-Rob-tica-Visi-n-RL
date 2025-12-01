# hand_config.py

import os

# tamaño de la ventana de la mano
PREVIEW_W = 640
PREVIEW_H = 480

# reconocimiento en tiempo real
RECOGNIZE_MODE = True
CONFIDENCE_THRESHOLD = 1.2  # ajusta según tu dataset

# si tienes calibración de la cámara de la mano
USE_UNDISTORT_HAND = False
HAND_CAMERA_PARAMS_PATH = "camera_params_hand.npz"  # cambia si lo llamas distinto

# carpeta donde se guardan los gestos
GESTURES_DIR = os.path.join(os.path.dirname(__file__), "gestures")

# carpeta donde se guardan las secuencias grabadas
SEQUENCES_DIR = os.path.join(os.path.dirname(__file__), "sequences")

# asegúrate de que existan
os.makedirs(GESTURES_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)