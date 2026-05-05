import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dario/Desktop/aerial_project/ros2_ws/install/aerial_grasping_control'
