cmake_minimum_required(VERSION 3.0.2)
project(vision_figuras)

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
  src/vision_figuras/figuras_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
