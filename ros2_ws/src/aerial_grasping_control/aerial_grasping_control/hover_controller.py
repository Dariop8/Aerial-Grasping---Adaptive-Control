import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion to roll, pitch, yaw.
    """
    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


class HoverController(Node):
    def __init__(self):
        super().__init__("hover_controller")

        # =========================
        # Drone physical parameters
        # =========================

        # Approximate total mass:
        # base 2.4 + link1 0.15 + link2 0.29 + gripper 0.10 + 4 rotors 0.12 = 3.06 kg
        self.mass = 3.06
        self.g = 9.81

        # Rotor arm length. Must match URDF/plugin rotor positions.
        self.l = 0.25

        # Yaw moment coefficient. Must match the motor_thrust_plugin.
        self.k_yaw = 0.02

        # Maximum thrust per motor in Newton.
        self.f_min = 0.0
        self.f_max = 15.0

        # Desired hover altitude.
        self.z_des = 1.5

        # =========================
        # Controller gains
        # =========================

        # Altitude PD gains.
        self.kp_z = 18.0
        self.kd_z = 10.0

        # Attitude PD gains.
        # These are intentionally conservative.
        self.kp_roll = 1.5
        self.kd_roll = 0.5

        self.kp_pitch = 1.5
        self.kd_pitch = 0.5

        self.kp_yaw = 0.4
        self.kd_yaw = 0.15

        # =========================
        # State variables
        # =========================

        self.have_odom = False

        self.z = 0.0
        self.z_dot = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        self.yaw_des = None

        self.last_log_time = self.get_clock().now()

        # =========================
        # ROS interfaces
        # =========================

        self.odom_sub = self.create_subscription(
            Odometry,
            "/drone/odom",
            self.odom_callback,
            10
        )

        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            "/motor_thrust_commands",
            10
        )

        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.get_logger().info("Hover controller started")
        self.get_logger().info("Waiting for /drone/odom...")

    def odom_callback(self, msg):
        # Position
        self.z = msg.pose.pose.position.z

        # Linear velocity
        self.z_dot = msg.twist.twist.linear.z

        # Orientation
        q = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = quaternion_to_euler(
            q.x, q.y, q.z, q.w
        )

        # Angular velocity.
        # For this first controller we use it directly as roll/pitch/yaw rate approximation.
        self.roll_rate = msg.twist.twist.angular.x
        self.pitch_rate = msg.twist.twist.angular.y
        self.yaw_rate = msg.twist.twist.angular.z

        if self.yaw_des is None:
            self.yaw_des = self.yaw
            self.get_logger().info(
                f"Initial yaw_des = {self.yaw_des:.3f} rad"
            )

        self.have_odom = True

    def mixer(self, F, Mx, My, Mz):
        """
        Convert total thrust and body moments into individual motor thrusts.

        Rotor positions:
          rotor 1: x=+l, y=+l
          rotor 2: x=-l, y=+l
          rotor 3: x=-l, y=-l
          rotor 4: x=+l, y=-l

        Force is along local +Z.
        Moment from vertical force is r x F:
          Mx = y * f
          My = -x * f

        Yaw directions are:
          [+1, -1, +1, -1]
        """

        B = np.array([
            [1.0,       1.0,       1.0,       1.0],
            [self.l,    self.l,   -self.l,   -self.l],
            [-self.l,   self.l,    self.l,   -self.l],
            [self.k_yaw, -self.k_yaw, self.k_yaw, -self.k_yaw],
        ])

        u = np.array([F, Mx, My, Mz])

        try:
            f = np.linalg.solve(B, u)
        except np.linalg.LinAlgError:
            self.get_logger().error("Mixer matrix is singular")
            f = np.zeros(4)

        f = np.clip(f, self.f_min, self.f_max)
        return f

    def control_loop(self):
        if not self.have_odom:
            return

        # =========================
        # Altitude control
        # =========================

        z_error = self.z_des - self.z
        z_dot_error = 0.0 - self.z_dot

        F = (
            self.mass * self.g
            + self.kp_z * z_error
            + self.kd_z * z_dot_error
        )

        # Do not command negative total thrust.
        F = max(0.0, F)

        # =========================
        # Attitude control
        # =========================

        roll_des = 0.0
        pitch_des = 0.0

        yaw_error = wrap_angle(self.yaw_des - self.yaw)

        Mx = (
            self.kp_roll * (roll_des - self.roll)
            + self.kd_roll * (0.0 - self.roll_rate)
        )

        My = (
            self.kp_pitch * (pitch_des - self.pitch)
            + self.kd_pitch * (0.0 - self.pitch_rate)
        )

        Mz = (
            self.kp_yaw * yaw_error
            + self.kd_yaw * (0.0 - self.yaw_rate)
        )

        # =========================
        # Motor allocation
        # =========================

        motor_thrusts = self.mixer(F, Mx, My, Mz)

        msg = Float64MultiArray()
        msg.data = motor_thrusts.tolist()
        self.motor_pub.publish(msg)

        # =========================
        # Logging
        # =========================

        now = self.get_clock().now()
        dt_log = (now - self.last_log_time).nanoseconds * 1e-9

        if dt_log > 1.0:
            self.last_log_time = now

            self.get_logger().info(
                f"z={self.z:.3f} z_des={self.z_des:.3f} "
                f"z_dot={self.z_dot:.3f} | "
                f"rpy=[{self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f}] | "
                f"F={F:.2f} M=[{Mx:.2f}, {My:.2f}, {Mz:.2f}] | "
                f"motors={np.round(motor_thrusts, 2)}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = HoverController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Send zero thrust on shutdown.
    zero_msg = Float64MultiArray()
    zero_msg.data = [0.0, 0.0, 0.0, 0.0]
    node.motor_pub.publish(zero_msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()