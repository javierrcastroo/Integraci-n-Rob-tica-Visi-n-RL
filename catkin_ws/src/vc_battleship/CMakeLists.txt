cmake_minimum_required(VERSION 3.0.2)
project(vc_battleship)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

# si vais a usar el service:
add_service_files(
  FILES
  ResolveCell.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    src/webcam_publisher.py
    src/usb_cam_publisher.py
    src/gesture_node.py
    src/board_node.py
    src/game_logic_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
