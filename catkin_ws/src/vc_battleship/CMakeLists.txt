cmake_minimum_required(VERSION 3.0.2)
project(vc_battleship)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
