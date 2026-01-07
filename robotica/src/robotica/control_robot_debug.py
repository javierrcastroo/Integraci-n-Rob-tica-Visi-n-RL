#!/usr/bin/env python3

import rospy
from control_robot import ControlRobot
from geometry_msgs.msg import Pose
import pprint

def main():
    rospy.init_node("control_robot_debug", anonymous=True)
    control = ControlRobot(init_ros_node=False)

    rospy.sleep(1.0)

    pose = control.pose_actual()

    print("\n\n===== POSE ACTUAL DEL ROBOT =====")
    pprint.pprint({
        "x": pose.position.x,
        "y": pose.position.y,
        "z": pose.position.z,
        "ox": pose.orientation.x,
        "oy": pose.orientation.y,
        "oz": pose.orientation.z,
        "ow": pose.orientation.w,
    })
    print("=================================\n\n")

    print("Ahora pruébalo moviendo articulaciones y vuelve a imprimir la pose.")

if __name__ == "__main__":
    main()
