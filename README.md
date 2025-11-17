# Integración Robótica Visión RL

chmod +x src/VC_Gesture/webcam_publisher.py

chmod +x src/VC_Gesture/gesture_node.py

(abrá mas)

rm -rf build/ devel/ log/ # limpia compilaciones previas 

catkin_make 

source devel/setup.bash 

rospack list | grep VC_Gesture

roslaunch vc_gesture gesture.launch

