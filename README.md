# Integración Robótica Visión RL

chmod +x src/VC_Gesture/src/VC_Gesture/hand_camera_node.py

chmod +x src/VC_Gesture/src/VC_Gesture/gesture_node.py

chmod +x src/VC_Board/src/VC_Board/board_camera_node.py

chmod +x src/VC_Board/src/VC_Board/board_node.py

chmod +x src/RB/src/RB/game_logic_node.py

chmod +x src/RL/src/RL/rl_agent_node.py

(abrá mas)


unzip src/VC_Gesture/src/VC_Gesture/gestures.zip -d src/VC_Gesture/src/VC_Gesture/


rm -rf build/ devel/ log/ # limpia compilaciones previas 

catkin_make 

source devel/setup.bash 


rospack list | grep -E "VC_Gesture|VC_Board|RL|RB"


roslaunch VC_Gesture gesture.launch

roslaunch VC_Board board.launch

roslaunch RB logic.launch

roslaunch RL rlModel.launch







