# storage.py
import os
import time
import json
import numpy as np
from hand_config import GESTURES_DIR

os.makedirs(GESTURES_DIR, exist_ok=True)

def save_gesture_example(feature_vec, label, save_dir=GESTURES_DIR):
    ts = int(time.time() * 1000)
    fp = os.path.join(save_dir, f"{label}_{ts}.npz")
    np.savez(fp, feature=feature_vec, label=label)
    print(f"[INFO] Gesto guardado en {fp}")

def load_gesture_gallery(save_dir=GESTURES_DIR):
    gallery = []
    for fname in os.listdir(save_dir):
        if not fname.endswith(".npz"):
            continue
        data = np.load(os.path.join(save_dir, fname), allow_pickle=True)
        feature = data["feature"]
        label = str(data["label"])
        gallery.append((feature, label))
    print(f"[INFO] Cargadas {len(gallery)} muestras en galería.")
    return gallery

def save_sequence_json(acciones, out_dir="gestures"):
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    payload = {
        "timestamp": ts,
        "sequence": acciones
    }
    fp = os.path.join(out_dir, f"sequence_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Secuencia guardada en {fp}")