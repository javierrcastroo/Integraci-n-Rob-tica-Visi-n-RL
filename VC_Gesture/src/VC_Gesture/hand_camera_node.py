#!/usr/bin/env python3
from usb_camera_node import main


if __name__ == "__main__":
    main(
        node_name="hand_camera_node",
        default_topic="hand_camera/image_raw",
        default_frame="hand_camera_optical_frame",
    )
