cmake_minimum_required(VERSION 3.0.2)
project(VC_Gesture)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/VC_Gesture/webcam_publisher.py
  src/VC_Gesture/gesture_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
