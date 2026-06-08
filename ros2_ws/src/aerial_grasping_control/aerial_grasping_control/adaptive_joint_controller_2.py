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
        self.q0 = None
        self.have_joint_state = False

        self.create_subscription(
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

        self.start_time = None
        self.last_log_time = 0.0
        self.current_phase = None

        self.Kp = np.diag([3.0, 1.2])
        self.Kd = np.diag([0.4, 0.25])

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('PD joint controller started')
        self.get_logger().info('Waiting for initial joint state...')

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

        if self.q0 is None:
            self.q0 = self.q.copy()
            self.start_time = self.get_clock().now()
            self.get_logger().info(f'Initial q0 = {np.round(self.q0, 3)}')

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
    #     q_home = self.q0
    #     q_goal = self.q0 + np.array([deg2rad(30.0), 0.0])

    #     if t < 2.0:
    #         return q_home, np.zeros(2), "INITIAL_HOLD"

    #     if t < 10.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_home,
    #             q_goal,
    #             t,
    #             t_start=2.0,
    #             duration=8.0
    #         )
    #         return q_des, q_dot_des, "MOVE_JOINT1_PLUS_30"

    #     return q_goal, np.zeros(2), "HOLD_GOAL"

    def desired_trajectory(self, t):
        q_home = self.q0.copy()

        # Target relativi alla posa iniziale.
        # Modifica questi valori se vuoi una traiettoria più/meno aggressiva.
        q_pre_grasp = q_home + np.array([
            deg2rad(30.0),   # joint1: +30 deg
            deg2rad(20.0)    # joint2: +20 deg
        ])

        q_grasp = q_home + np.array([
            deg2rad(30.0),   # joint1 resta a +30 deg
            deg2rad(-10.0)   # joint2 scende rispetto alla home
        ])

        q_retract = q_home + np.array([
            deg2rad(10.0),   # joint1 torna parzialmente indietro
            deg2rad(25.0)    # joint2 risale
        ])

        # 0–2 s: resta fermo nella configurazione iniziale
        if t < 2.0:
            return q_home, np.zeros(2), "INITIAL_HOLD"

        # 2–10 s: vai verso pre-grasp lentamente
        elif t < 10.0:
            q_des, q_dot_des = self.interpolate(
                q_home,
                q_pre_grasp,
                t,
                t_start=2.0,
                duration=8.0
            )
            return q_des, q_dot_des, "MOVE_TO_PRE_GRASP"

        # 10–14 s: vai verso grasp
        elif t < 14.0:
            q_des, q_dot_des = self.interpolate(
                q_pre_grasp,
                q_grasp,
                t,
                t_start=10.0,
                duration=4.0
            )
            return q_des, q_dot_des, "MOVE_TO_GRASP"

        # 14–17 s: mantieni grasp
        elif t < 17.0:
            return q_grasp, np.zeros(2), "GRASP_HOLD"

        # 17–23 s: retrazione
        elif t < 23.0:
            q_des, q_dot_des = self.interpolate(
                q_grasp,
                q_retract,
                t,
                t_start=17.0,
                duration=6.0
            )
            return q_des, q_dot_des, "RETRACT"

        # 23–29 s: torna alla home iniziale
        elif t < 35.0:
            q_des, q_dot_des = self.interpolate(
                q_retract,
                q_home,
                t,
                t_start=23.0,
                duration=6.0
            )
            return q_des, q_dot_des, "RETURN_HOME"

        # dopo 29 s: resta fermo alla home iniziale
        else:
            return q_home, np.zeros(2), "HOME_HOLD_FINAL"

    def control_loop(self):
        if not self.have_joint_state or self.q0 is None or self.start_time is None:
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9

        q_des, q_dot_des, phase = self.desired_trajectory(t)

        if phase != self.current_phase:
            self.current_phase = phase
            self.get_logger().info(f'===== NEW PHASE: {phase} at t={t:.2f}s =====')

        e = q_des - self.q
        e_dot = q_dot_des - self.q_dot

        tau = self.Kp @ e + self.Kd @ e_dot

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.command_pub.publish(msg)

        if t - self.last_log_time > 1.0:
            self.last_log_time = t
            self.get_logger().info(
                f'phase={phase} | t={t:.2f} | '
                f'q={np.round(self.q, 3)} | '
                f'q_des={np.round(q_des, 3)} | '
                f'e={np.round(e, 3)} | '
                f'tau={np.round(tau, 3)}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveJointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()