#!/usr/bin/env python3
from usb_camera_node import main


if __name__ == "__main__":
    main(
        node_name="board_camera_node",
        default_topic="board_camera/image_raw",
        default_frame="board_camera_optical_frame",
    )
