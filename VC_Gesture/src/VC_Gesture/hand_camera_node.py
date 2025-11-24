#!/usr/bin/env python3
import os
import sys

# --- AÑADIDO: para encontrar usb_camera_node.py real en scripts/ ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../scripts/hand
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)                    # .../scripts
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from usb_camera_node import main


if __name__ == "__main__":
    main(
        node_name="hand_camera_node",
        default_topic="hand_camera/image_raw",
        default_frame="hand_camera_optical_frame",
    )
