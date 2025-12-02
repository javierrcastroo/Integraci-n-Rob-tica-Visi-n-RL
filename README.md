# Integración Robótica Visión refuerzo

chmod +x src/vision_gestos/src/vision_gestos/hand_main.py

chmod +x src/vision_tablero/src/vision_tablero/board_main.py

chmod +x src/robotica/src/robotica/game_logic_node.py

chmod +x src/robotica/src/robotica/robot_attack_node.py

chmod +x src/refuerzo/src/refuerzo/rl_agent_node.py

(abrá mas)

chmod +x src/refuerzo/setup_rl_env.sh

./src/refuerzo/setup_rl_env.sh

---------------------------------------------------------------------------------------------


rm -rf build/ devel/ log/ # limpia compilaciones previas 

catkin_make 

source devel/setup.bash 

rospack list | grep -E "vision_gestos|vision_tablero|refuerzo|robotica"

---------------------------------------------------------------------------------------------

roslaunch robotica sistema.launch

roslaunch vision_gestos gesture.launch

roslaunch vision_tablero board.launch

roslaunch robotica logic.launch

roslaunch refuerzo rlModel.launch



