import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class AdaptiveHoverController(Node):
    def __init__(self):
        super().__init__("adaptive_hover_controller")

        # =========================
        # Physical parameters
        # =========================

        self.mass = 3.06
        self.g = 9.81

        self.l = 0.25
        self.k_yaw = 0.02

        self.f_min = 0.0
        self.f_max = 15.0

        # Desired hover altitude
        self.z_des = 1.5

        # =========================
        # Adaptive/robust controller gains
        # =========================

        # State chi = [z, roll, pitch, yaw]
        #
        # e = chi - chi_des
        # s = e_dot + Phi e
        # u = - Lambda s - adaptive_scale * rho * s_direction

        self.Phi = np.diag([
            1.2,   # z
            2.0,   # roll
            2.0,   # pitch
            1.0    # yaw
        ])

        self.Lambda = np.diag([
            8.0,   # z correction
            1.5,   # roll moment
            1.5,   # pitch moment
            0.4    # yaw moment
        ])

        # Adaptive gains K_hat_i
        self.K_hat = np.array([0.01, 0.01, 0.01], dtype=float)

        # Leakage terms. Higher values keep adaptation bounded.
        self.nu = np.array([8.0, 12.0, 12.0], dtype=float)

        # Smooth approximation parameter
        self.delta = 0.1

        # Scale the adaptive term at the beginning.
        # Start small; increase later only if stable.
        self.adaptive_scale = 0.05

        # Saturation for virtual control:
        # u_virtual = [F_correction, Mx, My, Mz]
        self.F_correction_limit = 12.0
        self.Mx_limit = 2.0
        self.My_limit = 2.0
        self.Mz_limit = 0.8

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

        self.last_time = None
        self.last_log_time = None

        # =========================
        # ROS interfaces
        # =========================

        self.create_subscription(
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

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Adaptive hover controller started")
        self.get_logger().info("Waiting for /drone/odom...")

    def odom_callback(self, msg):
        self.z = msg.pose.pose.position.z
        self.z_dot = msg.twist.twist.linear.z

        q = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = quaternion_to_euler(
            q.x, q.y, q.z, q.w
        )

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
        Convert [F, Mx, My, Mz] into [f1, f2, f3, f4].

        Rotor positions:
            rotor 1: x=+l, y=+l
            rotor 2: x=-l, y=+l
            rotor 3: x=-l, y=-l
            rotor 4: x=+l, y=-l

        The plugin uses yaw directions:
            [+1, -1, +1, -1]
        """

        B = np.array([
            [1.0,        1.0,        1.0,        1.0],
            [self.l,     self.l,    -self.l,    -self.l],
            [-self.l,    self.l,     self.l,    -self.l],
            [self.k_yaw, -self.k_yaw, self.k_yaw, -self.k_yaw],
        ])

        u = np.array([F, Mx, My, Mz])

        try:
            f = np.linalg.solve(B, u)
        except np.linalg.LinAlgError:
            self.get_logger().error("Mixer matrix is singular")
            f = np.zeros(4)

        return np.clip(f, self.f_min, self.f_max)

    def compute_adaptive_control(self, chi, chi_dot, chi_des, chi_dot_des, dt):
        """
        Adaptive robust law inspired by the paper.

        e = chi - chi_des
        e_dot = chi_dot - chi_dot_des
        s = e_dot + Phi e
        xi = [e, e_dot]

        K_hat_dot_i = ||s|| ||xi||^i - nu_i K_hat_i

        rho = K0 + K1 ||xi|| + K2 ||xi||²

        u = - Lambda s - adaptive_scale * rho * s / sqrt(||s||² + delta)
        """

        e = chi - chi_des
        e[3] = wrap_angle(e[3])

        e_dot = chi_dot - chi_dot_des

        s = e_dot + self.Phi @ e

        xi = np.concatenate((e, e_dot))
        xi_norm = np.linalg.norm(xi)
        s_norm = np.linalg.norm(s)

        K_dot = np.array([
            s_norm - self.nu[0] * self.K_hat[0],
            s_norm * xi_norm - self.nu[1] * self.K_hat[1],
            s_norm * xi_norm**2 - self.nu[2] * self.K_hat[2],
        ])

        self.K_hat += K_dot * dt
        self.K_hat = np.maximum(self.K_hat, 0.0)

        rho = (
            self.K_hat[0]
            + self.K_hat[1] * xi_norm
            + self.K_hat[2] * xi_norm**2
        )

        s_direction = s / math.sqrt(s_norm**2 + self.delta)

        u = -self.Lambda @ s - self.adaptive_scale * rho * s_direction

        return u, e, s, rho

    def control_loop(self):
        if not self.have_odom or self.yaw_des is None:
            return

        now = self.get_clock().now()

        if self.last_time is None:
            self.last_time = now
            self.last_log_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.1:
            self.get_logger().warn(f"Invalid dt={dt:.4f}, skipping control step")
            return

        # =========================
        # State vector
        # =========================

        chi = np.array([
            self.z,
            self.roll,
            self.pitch,
            self.yaw
        ])

        chi_dot = np.array([
            self.z_dot,
            self.roll_rate,
            self.pitch_rate,
            self.yaw_rate
        ])

        chi_des = np.array([
            self.z_des,
            0.0,
            0.0,
            self.yaw_des
        ])

        chi_dot_des = np.zeros(4)

        # =========================
        # Adaptive robust control
        # =========================

        u_virtual, e, s, rho = self.compute_adaptive_control(
            chi,
            chi_dot,
            chi_des,
            chi_dot_des,
            dt
        )

        F_correction = np.clip(
            u_virtual[0],
            -self.F_correction_limit,
            self.F_correction_limit
        )

        Mx = np.clip(u_virtual[1], -self.Mx_limit, self.Mx_limit)
        My = np.clip(u_virtual[2], -self.My_limit, self.My_limit)
        Mz = np.clip(u_virtual[3], -self.Mz_limit, self.Mz_limit)

        # Add gravity compensation
        F = self.mass * self.g + F_correction
        F = max(0.0, F)

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

        dt_log = (now - self.last_log_time).nanoseconds * 1e-9

        if dt_log > 1.0:
            self.last_log_time = now

            self.get_logger().info(
                f"z={self.z:.3f} z_des={self.z_des:.3f} "
                f"z_dot={self.z_dot:.3f} | "
                f"rpy=[{self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f}] | "
                f"e={np.round(e, 3)} | "
                f"s={np.round(s, 3)} | "
                f"rho={rho:.3f} K_hat={np.round(self.K_hat, 3)} | "
                f"F={F:.2f} M=[{Mx:.2f}, {My:.2f}, {Mz:.2f}] | "
                f"motors={np.round(motor_thrusts, 2)}"
            )

    def stop_motors(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = AdaptiveHoverController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_motors()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()