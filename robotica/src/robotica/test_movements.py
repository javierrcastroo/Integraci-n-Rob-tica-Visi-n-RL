from control_robot import ControlRobot

control = ControlRobot()

pose_actual = control.pose_actual()
pose_actual.position.z -= 0.05

control.mover_trayectoria([pose_actual])