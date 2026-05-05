import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def deg2rad(deg):
    return deg * math.pi / 180.0


def smoothstep(s):
    return 3.0 * s**2 - 2.0 * s**3


def smoothstep_dot(s):
    return 6.0 * s - 6.0 * s**2


class AdaptiveJointController(Node):
    def __init__(self):
        super().__init__('adaptive_joint_controller')

        self.joint_names = ['joint1', 'joint2']

        self.q = np.zeros(2)
        self.q_dot = np.zeros(2)
        self.have_joint_state = False

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_effort_controller/commands',
            10
        )

        self.start_time = self.get_clock().now()
        self.last_time = self.start_time

        # Controller gains.
        # These are the simplified Phi and Lambda matrices from the paper.
        self.Phi = np.diag([0.6, 0.6]) #self.Phi = np.diag([1.2, 1.2])
        self.Lambda = np.diag([1.0, 1.0]) #self.Lambda = np.diag([3.0, 3.0])

        # Adaptive gains K_hat_i.
        self.K_hat = np.array([0.01, 0.01, 0.01], dtype=float) #self.K_hat = np.array([0.1, 0.1, 0.1], dtype=float)

        # Leakage terms nu_i from the adaptive law.
        self.nu = np.array([2.0, 5.0, 5.0], dtype=float)

        # Smooth approximation parameter.
        self.delta = 0.1

        # Torque saturation for safety.
        self.tau_limit = 2.0

        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.get_logger().info('Adaptive joint controller started')

        self.current_phase = None

    def joint_state_callback(self, msg):
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in name_to_index:
                return

            idx = name_to_index[joint_name]

            if len(msg.position) > idx:
                self.q[i] = msg.position[idx]

            if len(msg.velocity) > idx:
                self.q_dot[i] = msg.velocity[idx]

        self.have_joint_state = True

    def interpolate(self, q_start, q_goal, t, t_start, duration):
        if t <= t_start:
            return q_start, np.zeros(2)

        if t >= t_start + duration:
            return q_goal, np.zeros(2)

        s = (t - t_start) / duration
        sigma = smoothstep(s)
        sigma_dot = smoothstep_dot(s) / duration

        q_des = q_start + sigma * (q_goal - q_start)
        q_dot_des = sigma_dot * (q_goal - q_start)

        return q_des, q_dot_des

    # def desired_trajectory(self, t):
    #     q_home = np.array([deg2rad(0.0), deg2rad(0.0)])
    #     q_pre_grasp = np.array([deg2rad(45.0), deg2rad(45.0)])
    #     q_grasp = np.array([deg2rad(45.0), deg2rad(15.0)])
    #     q_retract = np.array([deg2rad(45.0), deg2rad(60.0)])

    #     if t < 2.0:
    #         return q_home, np.zeros(2), "HOME_HOLD"

    #     elif t < 6.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_home,
    #             q_pre_grasp,
    #             t,
    #             t_start=2.0,
    #             duration=4.0
    #         )
    #         return q_des, q_dot_des, "MOVE_TO_PRE_GRASP"

    #     elif t < 8.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_pre_grasp,
    #             q_grasp,
    #             t,
    #             t_start=6.0,
    #             duration=2.0
    #         )
    #         return q_des, q_dot_des, "MOVE_TO_GRASP"

    #     elif t < 10.0:
    #         return q_grasp, np.zeros(2), "GRASP_HOLD"

    #     elif t < 13.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_grasp,
    #             q_retract,
    #             t,
    #             t_start=10.0,
    #             duration=3.0
    #         )
    #         return q_des, q_dot_des, "RETRACT"

    #     else:
    #         return q_retract, np.zeros(2), "RETRACT_HOLD"

    def desired_trajectory(self, t):
        q_home = np.array([deg2rad(0.0), deg2rad(0.0)])
        q_test = np.array([deg2rad(10.0), deg2rad(0.0)])

        if t < 2.0:
            return q_home, np.zeros(2), "HOME_HOLD"

        elif t < 6.0:
            q_des, q_dot_des = self.interpolate(
                q_home,
                q_test,
                t,
                t_start=2.0,
                duration=4.0
            )
            return q_des, q_dot_des, "MOVE_JOINT1_POSITIVE"

        elif t < 10.0:
            return q_test, np.zeros(2), "HOLD_JOINT1_POSITIVE"

        elif t < 14.0:
            q_des, q_dot_des = self.interpolate(
                q_test,
                q_home,
                t,
                t_start=10.0,
                duration=4.0
            )
            return q_des, q_dot_des, "RETURN_HOME"

        else:
            return q_home, np.zeros(2), "HOME_HOLD_FINAL"

    def control_loop(self):
        if not self.have_joint_state:
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.1:
            return

        # q_des, q_dot_des = self.desired_trajectory(t)
        q_des, q_dot_des, phase = self.desired_trajectory(t)

        if phase != self.current_phase:
            self.current_phase = phase
            self.get_logger().info(f'===== NEW PHASE: {phase} at t={t:.2f}s =====')

        # Paper convention:
        # e = chi - chi_d
        # Here chi = q.
        e = self.q - q_des
        e_dot = self.q_dot - q_dot_des

        s = e_dot + self.Phi @ e

        xi = np.concatenate((e, e_dot))
        xi_norm = np.linalg.norm(xi)
        s_norm = np.linalg.norm(s)

        # Adaptive law:
        # K_hat_dot_i = ||s|| ||xi||^i - nu_i K_hat_i
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

        # Smooth version of s / ||s|| from Remark 1 in the paper.
        s_direction = s / math.sqrt(s_norm**2 + self.delta)

        # tau = -self.Lambda @ s - rho * s_direction
        tau = self.Lambda @ s + rho * s_direction

        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.command_pub.publish(msg)

        # if int(t * 10) % 20 == 0:
        #     self.get_logger().info(
        #         f't={t:.2f} q={self.q} q_des={q_des} tau={tau} K_hat={self.K_hat}'
        #     )

        if not hasattr(self, 'last_log_time'):
            self.last_log_time = 0.0

        if t - self.last_log_time > 1.0:
            self.last_log_time = t
            # self.get_logger().info(
            #     f't={t:.2f} q={self.q} q_des={q_des} tau={tau} K_hat={self.K_hat}'
            # )
            error_norm = np.linalg.norm(e)

            self.get_logger().info(
                f'phase={phase} | t={t:.2f} | '
                f'q={np.round(self.q, 3)} | '
                f'q_des={np.round(q_des, 3)} | '
                f'e_norm={error_norm:.3f} | '
                f'tau={np.round(tau, 3)} | '
                f'K_hat={np.round(self.K_hat, 3)}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = AdaptiveJointController()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()