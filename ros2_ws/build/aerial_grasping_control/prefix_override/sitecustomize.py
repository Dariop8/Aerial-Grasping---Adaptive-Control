import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dario/Desktop/Aerial-Grasping---Adaptive-Control/ros2_ws/install/aerial_grasping_control'
