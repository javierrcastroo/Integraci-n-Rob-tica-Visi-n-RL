cmake_minimum_required(VERSION 3.0.2)
project(vision_gestos)

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
  src/vision_gestos/hand_main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
