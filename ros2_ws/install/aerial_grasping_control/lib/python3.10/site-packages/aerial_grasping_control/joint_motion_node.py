import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointMotionNode(Node):
    def __init__(self):
        super().__init__('joint_motion_node')

        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(
            0.02,
            self.timer_callback
        )

        self.get_logger().info('Joint motion node started')

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9

        q1 = 0.6 * math.sin(0.7 * t)
        q2 = 0.8 * math.sin(0.5 * t)

        q1_dot = 0.6 * 0.7 * math.cos(0.7 * t)
        q2_dot = 0.8 * 0.5 * math.cos(0.5 * t)

        msg = JointState()
        msg.header.stamp = now.to_msg()

        msg.name = ['joint1', 'joint2']
        msg.position = [q1, q2]
        msg.velocity = [q1_dot, q2_dot]
        msg.effort = [0.0, 0.0]

        self.joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = JointMotionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()