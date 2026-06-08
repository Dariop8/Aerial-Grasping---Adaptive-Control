import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def deg2rad(deg):
    return deg * math.pi / 180.0


def smoothstep(s):
    return 3.0 * s**2 - 2.0 * s**3


def smoothstep_dot(s):
    return 6.0 * s - 6.0 * s**2


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


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


class FullAdaptiveController(Node):
    def __init__(self):
        super().__init__("full_adaptive_controller")

        # ==========================================================
        # Physical parameters of simulated UAM
        # ==========================================================

        self.mass = 3.06
        self.g = 9.81

        # Quadrotor geometry used by the motor mixer
        self.l = 0.25
        self.k_yaw = 0.02

        self.f_min = 0.0
        self.f_max = 15.0

        # ==========================================================
        # Desired drone trajectory
        #
        # For now:
        # - hold initial x, y, z
        # - hold initial yaw
        #
        # This corresponds to testing the paper controller while the
        # manipulator executes a task.
        # ==========================================================

        self.x_des = None
        self.y_des = None
        self.z_des = None
        self.yaw_des = None

        # ==========================================================
        # PAPER CONTROL PARAMETERS
        #
        # From the experimental section of the paper:
        #
        # Phi = diag{1.0, 1.0, 1.2, 1.1, 1.1, 1.0, 1.2, 1.2}
        # Lambda = diag{2.0, 2.0, 3.5, 1.5, 1.5, 1.2, 3.0, 3.0}
        # K_hat_i(0) = 0.1
        # nu0 = 2.0, nu1 = nu2 = 5.0
        # delta = 0.1
        #
        # chi = [x, y, z, roll, pitch, yaw, q1, q2]
        # ==========================================================

        # self.Phi = np.diag([
        #     1.0,   # x
        #     1.0,   # y
        #     1.2,   # z
        #     1.1,   # roll
        #     1.1,   # pitch
        #     1.0,   # yaw
        #     1.2,   # joint1
        #     1.2    # joint2
        # ])

        # self.Lambda = np.diag([
        #     2.0,   # x generalized force
        #     2.0,   # y generalized force
        #     3.5,   # z generalized force
        #     1.5,   # roll moment
        #     1.5,   # pitch moment
        #     1.2,   # yaw moment
        #     3.0,   # joint1 torque
        #     3.0    # joint2 torque
        # ])

        # ==========================================================
        # PAPER-LIKE CONTROL PARAMETERS - STABILIZED FOR GAZEBO
        # chi = [x, y, z, roll, pitch, yaw, q1, q2]
        # ==========================================================

        self.Phi = np.diag([
            1.40,   # x
            1.40,   # y
            1.40,   # z
            3.5,    # roll
            3.5,    # pitch
            1.5,    # yaw
            1.0,    # joint1
            0.6     # joint2
        ])

        self.Lambda = np.diag([
            6.00,   # ax virtual command
            6.00,   # ay virtual command
            9.00,   # az virtual command
            2.2,    # roll-related virtual moment
            2.2,    # pitch-related virtual moment
            0.7,    # yaw-related virtual moment
            1.0,    # tau1
            0.6     # tau2
        ])

        self.K_hat = np.array([0.01, 0.01, 0.01], dtype=float)
        self.nu = np.array([10.0, 14.0, 14.0], dtype=float)

        self.delta = 0.3

        # ==========================================================
        # Practical actuator limits
        #
        # These are not part of the theorem, but are necessary in
        # simulation because Gazebo motors and joint actuators saturate.
        # ==========================================================

        self.ax_limit = 4.0
        self.ay_limit = 4.0
        self.az_limit = 10.0

        self.max_tilt_des = deg2rad(15.0)

        self.Mx_limit = 2.2
        self.My_limit = 2.2
        self.Mz_limit = 0.8

        self.tau_arm_limit = 20.0

        # Optional numerical protection.
        # Set False if you want the purest possible implementation.
        # I recommend True for Gazebo.
        self.use_adaptive_caps = True
        self.K_hat_limit = 2.5
        self.rho_limit = 3.0

        # ==========================================================
        # Inner attitude adaptive stabilizer
        #
        # Reason:
        # The paper law produces generalized tau.
        # In a real/simulated quadrotor, x/y forces are not directly
        # actuated. They must be realized by desired roll/pitch and
        # then by motor moments.
        #
        # This block uses the same paper law form on:
        # chi_att = [roll, pitch, yaw]
        # ==========================================================

        self.Phi_att = np.diag([
            3.5,
            3.5,
            1.5
        ])

        self.Lambda_att = np.diag([
            2.2,
            2.2,
            0.7
        ])

        self.K_hat_att = np.array([0.01, 0.01, 0.01], dtype=float)
        self.nu_att = np.array([10.0, 14.0, 14.0], dtype=float)

        # ==========================================================
        # Arm state
        # ==========================================================

        self.joint_names = ["joint1", "joint2"]

        self.q = np.zeros(2)
        self.q_dot = np.zeros(2)
        self.q0 = None

        # ==========================================================
        # Drone state
        # ==========================================================

        self.have_odom = False
        self.have_joint_state = False

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.x_dot = 0.0
        self.y_dot = 0.0
        self.z_dot = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        self.start_time = None
        self.last_time = None
        self.last_log_time = None

        # ==========================================================
        # ROS interfaces
        # ==========================================================

        self.create_subscription(
            Odometry,
            "/drone/odom",
            self.odom_callback,
            10
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            "/motor_thrust_commands",
            10
        )

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            "/arm_effort_controller/commands",
            10
        )

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Paper adaptive controller started")
        self.get_logger().info("Waiting for /drone/odom and /joint_states...")

    # ==========================================================
    # Callbacks
    # ==========================================================

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        self.x_dot = msg.twist.twist.linear.x
        self.y_dot = msg.twist.twist.linear.y
        self.z_dot = msg.twist.twist.linear.z

        q = msg.pose.pose.orientation

        self.roll, self.pitch, self.yaw = quaternion_to_euler(
            q.x,
            q.y,
            q.z,
            q.w
        )

        self.roll_rate = msg.twist.twist.angular.x
        self.pitch_rate = msg.twist.twist.angular.y
        self.yaw_rate = msg.twist.twist.angular.z

        if self.x_des is None:
            self.x_des = self.x
            self.y_des = self.y
            self.z_des = self.z

            self.get_logger().info(
                f"Initial position hold: "
                f"x_des={self.x_des:.3f}, "
                f"y_des={self.y_des:.3f}, "
                f"z_des={self.z_des:.3f}"
            )

        if self.yaw_des is None:
            self.yaw_des = self.yaw

            self.get_logger().info(
                f"Initial yaw_des = {self.yaw_des:.3f} rad"
            )

        self.have_odom = True

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

            self.get_logger().info(
                f"Initial arm q0 = {np.round(self.q0, 3)}"
            )

        self.have_joint_state = True

    # ==========================================================
    # Desired manipulator trajectory
    # ==========================================================

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

    def interpolate_scalar(self, start, goal, t, t_start, duration):
        if t <= t_start:
            return start, 0.0

        if t >= t_start + duration:
            return goal, 0.0

        s = (t - t_start) / duration
        sigma = smoothstep(s)
        sigma_dot = smoothstep_dot(s) / duration

        value = start + sigma * (goal - start)
        value_dot = sigma_dot * (goal - start)

        return value, value_dot

    def desired_drone_trajectory(self, t):
        x0 = self.x_des
        y0 = self.y_des
        z0 = self.z_des
        yaw0 = self.yaw_des

        x_goal = x0 + 0.6
        y_goal = y0
        z_goal = z0
        yaw_goal = yaw0

        # 0–6 s: hover iniziale
        if t < 6.0:
            return (
                np.array([x0, y0, z0, yaw0]),
                np.zeros(4),
                "DRONE_INITIAL_HOVER"
            )

        # 6–22 s: traslazione lenta in x
        elif t < 22.0:
            x_des, x_dot_des = self.interpolate_scalar(
                x0,
                x_goal,
                t,
                t_start=6.0,
                duration=16.0
            )

            return (
                np.array([x_des, y0, z0, yaw0]),
                np.array([x_dot_des, 0.0, 0.0, 0.0]),
                "DRONE_MOVE_X_PLUS"
            )

        # 22–75 s: hold nella nuova posizione mentre il braccio lavora
        elif t < 75.0:
            return (
                np.array([x_goal, y_goal, z_goal, yaw_goal]),
                np.zeros(4),
                "DRONE_HOLD_WORKSPACE"
            )

        # 75–91 s: ritorno alla posizione iniziale
        elif t < 91.0:
            x_des, x_dot_des = self.interpolate_scalar(
                x_goal,
                x0,
                t,
                t_start=75.0,
                duration=16.0
            )

            return (
                np.array([x_des, y0, z0, yaw0]),
                np.array([x_dot_des, 0.0, 0.0, 0.0]),
                "DRONE_RETURN_HOME"
            )

        else:
            return (
                np.array([x0, y0, z0, yaw0]),
                np.zeros(4),
                "DRONE_FINAL_HOVER"
            )

    # def desired_arm_trajectory(self, t):
    #     q_home = self.q0.copy()

    #     q_pre_grasp = q_home + np.array([
    #         deg2rad(25.0),
    #         deg2rad(10.0)
    #     ])

    #     q_grasp = q_home + np.array([
    #         deg2rad(25.0),
    #         deg2rad(-5.0)
    #     ])

    #     q_retract = q_home + np.array([
    #         deg2rad(5.0),
    #         deg2rad(15.0)
    #     ])

    #     if t < 5.0:
    #         return q_home, np.zeros(2), "ARM_INITIAL_HOLD"

    #     elif t < 17.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_home,
    #             q_pre_grasp,
    #             t,
    #             t_start=5.0,
    #             duration=12.0
    #         )

    #         return q_des, q_dot_des, "ARM_MOVE_TO_PRE_GRASP"

    #     elif t < 25.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_pre_grasp,
    #             q_grasp,
    #             t,
    #             t_start=17.0,
    #             duration=8.0
    #         )

    #         return q_des, q_dot_des, "ARM_MOVE_TO_GRASP"

    #     elif t < 29.0:
    #         return q_grasp, np.zeros(2), "ARM_GRASP_HOLD"

    #     elif t < 37.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_grasp,
    #             q_retract,
    #             t,
    #             t_start=29.0,
    #             duration=8.0
    #         )

    #         return q_des, q_dot_des, "ARM_RETRACT"

    #     elif t < 45.0:
    #         q_des, q_dot_des = self.interpolate(
    #             q_retract,
    #             q_home,
    #             t,
    #             t_start=37.0,
    #             duration=8.0
    #         )

    #         return q_des, q_dot_des, "ARM_RETURN_HOME"

    #     else:
    #         return q_home, np.zeros(2), "ARM_HOME_HOLD_FINAL"

    def desired_arm_trajectory(self, t):
        arm_start = 22.0
        q_home = self.q0.copy()

        # Task più visibile:
        # pre_grasp: braccio abbastanza avanzato
        # grasp: secondo giunto scende molto
        # retract: braccio si richiude e risale
        q_pre_grasp = q_home + np.array([
            deg2rad(35.0),
            deg2rad(18.0)
        ])

        q_grasp = q_home + np.array([
            deg2rad(42.0),
            deg2rad(-18.0)
        ])

        q_retract = q_home + np.array([
            deg2rad(12.0),
            deg2rad(28.0)
        ])

        # 0–6 s: stabilizzazione iniziale
        if t < arm_start + 6.0:
            return q_home, np.zeros(2), "ARM_INITIAL_HOLD"

        # 6–22 s: movimento evidente verso pre-grasp
        elif t < arm_start + 22.0:
            q_des, q_dot_des = self.interpolate(
                q_home,
                q_pre_grasp,
                t,
                t_start=arm_start + 6.0,
                duration=16.0
            )
            return q_des, q_dot_des, "ARM_MOVE_TO_PRE_GRASP"

        # 22–36 s: movimento evidente verso grasp
        elif t < arm_start + 36.0:
            q_des, q_dot_des = self.interpolate(
                q_pre_grasp,
                q_grasp,
                t,
                t_start=arm_start + 22.0,
                duration=14.0
            )
            return q_des, q_dot_des, "ARM_MOVE_TO_GRASP"

        # 36–43 s: hold in grasp
        elif t < arm_start + 43.0:
            return q_grasp, np.zeros(2), "ARM_GRASP_HOLD"

        # 43–57 s: retrazione
        elif t < arm_start + 57.0:
            q_des, q_dot_des = self.interpolate(
                q_grasp,
                q_retract,
                t,
                t_start=arm_start + 43.0,
                duration=14.0
            )
            return q_des, q_dot_des, "ARM_RETRACT"

        # 57–73 s: ritorno alla home
        elif t < arm_start + 73.0:
            q_des, q_dot_des = self.interpolate(
                q_retract,
                q_home,
                t,
                t_start=arm_start + 57.0,
                duration=16.0
            )
            return q_des, q_dot_des, "ARM_RETURN_HOME"

        else:
            return q_home, np.zeros(2), "ARM_HOME_HOLD_FINAL"
        
    # ==========================================================
    # Paper adaptive law
    # ==========================================================

    def paper_adaptive_law(
        self,
        chi,
        chi_dot,
        chi_des,
        chi_dot_des,
        Phi,
        Lambda,
        K_hat,
        nu,
        dt,
        yaw_index=None
    ):
        # e = chi - chi_d
        e = chi - chi_des

        if yaw_index is not None:
            e[yaw_index] = wrap_angle(e[yaw_index])

        # e_dot = chi_dot - chi_dot_d
        e_dot = chi_dot - chi_dot_des

        # s = e_dot + Phi e
        s = e_dot + Phi @ e

        # xi = [e^T, e_dot^T]^T
        xi = np.concatenate((e, e_dot))

        xi_norm = np.linalg.norm(xi)
        s_norm = np.linalg.norm(s)

        # K_hat_dot_i = ||s|| ||xi||^i - nu_i K_hat_i
        K_dot = np.array([
            s_norm - nu[0] * K_hat[0],
            s_norm * xi_norm - nu[1] * K_hat[1],
            s_norm * xi_norm**2 - nu[2] * K_hat[2],
        ])

        K_hat += K_dot * dt
        K_hat[:] = np.maximum(K_hat, 0.0)

        if self.use_adaptive_caps:
            K_hat[:] = np.clip(K_hat, 0.0, self.K_hat_limit)

        # rho = K0 + K1 ||xi|| + K2 ||xi||^2
        rho = (
            K_hat[0]
            + K_hat[1] * xi_norm
            + K_hat[2] * xi_norm**2
        )

        if self.use_adaptive_caps:
            rho = float(np.clip(rho, 0.0, self.rho_limit))

        # Remark 1 smooth version of s / ||s||
        s_direction = s / math.sqrt(s_norm**2 + self.delta)

        # tau = -Lambda s - rho * smooth(s)
        tau = -Lambda @ s - rho * s_direction

        return tau, e, e_dot, s, rho, xi_norm

    # ==========================================================
    # Mixer
    # ==========================================================

    def mixer(self, F, Mx, My, Mz):
        B = np.array([
            [1.0,         1.0,         1.0,         1.0],
            [self.l,      self.l,     -self.l,     -self.l],
            [-self.l,     self.l,      self.l,     -self.l],
            [self.k_yaw, -self.k_yaw,  self.k_yaw, -self.k_yaw],
        ])

        generalized = np.array([
            F,
            Mx,
            My,
            Mz
        ])

        try:
            motor_forces = np.linalg.solve(B, generalized)
        except np.linalg.LinAlgError:
            self.get_logger().error("Mixer matrix is singular")
            motor_forces = np.zeros(4)

        return np.clip(motor_forces, self.f_min, self.f_max)

    # ==========================================================
    # Main loop
    # ==========================================================

    def control_loop(self):
        if not self.have_odom:
            return

        if (
            self.x_des is None
            or self.y_des is None
            or self.z_des is None
            or self.yaw_des is None
        ):
            return

        now = self.get_clock().now()

        if self.start_time is None:
            self.start_time = now
            self.last_time = now
            self.last_log_time = now

            self.get_logger().info("Controller time initialized")

            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.1:
            self.get_logger().warn(f"Invalid dt={dt:.4f}, skipping")
            return

        t = (now - self.start_time).nanoseconds * 1e-9

        # ======================================================
        # Desired arm trajectory
        # ======================================================

        if self.have_joint_state and self.q0 is not None:
            q_des, q_dot_des, arm_phase = self.desired_arm_trajectory(t)
        else:
            q_des = self.q.copy()
            q_dot_des = np.zeros(2)
            arm_phase = "ARM_WAITING_FOR_JOINT_STATES"

        # ======================================================
        # Full paper state vector
        #
        # chi = [x, y, z, roll, pitch, yaw, q1, q2]
        # ======================================================

        chi = np.array([
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
            self.q[0],
            self.q[1]
        ])

        chi_dot = np.array([
            self.x_dot,
            self.y_dot,
            self.z_dot,
            self.roll_rate,
            self.pitch_rate,
            self.yaw_rate,
            self.q_dot[0],
            self.q_dot[1]
        ])

        # chi_des = np.array([
        #     self.x_des,
        #     self.y_des,
        #     self.z_des,
        #     0.0,
        #     0.0,
        #     self.yaw_des,
        #     q_des[0],
        #     q_des[1]
        # ])

        # chi_dot_des = np.array([
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     q_dot_des[0],
        #     q_dot_des[1]
        # ])

        drone_des, drone_dot_des, drone_phase = self.desired_drone_trajectory(t)

        chi_des = np.array([
            drone_des[0],       # x_des
            drone_des[1],       # y_des
            drone_des[2],       # z_des
            0.0,                # roll_des nominale
            0.0,                # pitch_des nominale
            drone_des[3],       # yaw_des
            q_des[0],
            q_des[1]
        ])

        chi_dot_des = np.array([
            drone_dot_des[0],   # x_dot_des
            drone_dot_des[1],   # y_dot_des
            drone_dot_des[2],   # z_dot_des
            0.0,
            0.0,
            drone_dot_des[3],   # yaw_dot_des
            q_dot_des[0],
            q_dot_des[1]
        ])

        tau, e, e_dot, s, rho, xi_norm = self.paper_adaptive_law(
            chi=chi,
            chi_dot=chi_dot,
            chi_des=chi_des,
            chi_dot_des=chi_dot_des,
            Phi=self.Phi,
            Lambda=self.Lambda,
            K_hat=self.K_hat,
            nu=self.nu,
            dt=dt,
            yaw_index=5
        )

        # ======================================================
        # Extract paper generalized input
        #
        # tau = [tau_x, tau_y, tau_z, tau_roll, tau_pitch,
        #        tau_yaw, tau_q1, tau_q2]
        # ======================================================

        tau_p = tau[0:3]
        tau_q = tau[3:6]
        tau_alpha = tau[6:8]

        # ======================================================
        # Position part:
        #
        # The paper writes tau_p = R^W_B U, U=[0,0,u1]^T.
        #
        # In this simulation we realize tau_x/tau_y indirectly:
        # tau_x, tau_y -> desired pitch/roll.
        #
        # tau_z changes total thrust around hover.
        # ======================================================

        ax_cmd = np.clip(tau_p[0], -self.ax_limit, self.ax_limit)
        ay_cmd = np.clip(tau_p[1], -self.ay_limit, self.ay_limit)
        az_cmd = np.clip(tau_p[2], -self.az_limit, self.az_limit)

        pitch_des = ax_cmd / self.g
        roll_des = -ay_cmd / self.g

        roll_des = np.clip(
            roll_des,
            -self.max_tilt_des,
            self.max_tilt_des
        )

        pitch_des = np.clip(
            pitch_des,
            -self.max_tilt_des,
            self.max_tilt_des
        )

        # ======================================================
        # Attitude realization
        #
        # tau_q from the paper is a generalized attitude command.
        # To make the motors track the desired roll/pitch/yaw,
        # we use the same adaptive law on attitude error.
        # ======================================================

        chi_att = np.array([
            self.roll,
            self.pitch,
            self.yaw
        ])

        chi_dot_att = np.array([
            self.roll_rate,
            self.pitch_rate,
            self.yaw_rate
        ])

        chi_des_att = np.array([
            roll_des,
            pitch_des,
            self.yaw_des
        ])

        chi_dot_des_att = np.zeros(3)

        tau_att, e_att, e_dot_att, s_att, rho_att, xi_norm_att = self.paper_adaptive_law(
            chi=chi_att,
            chi_dot=chi_dot_att,
            chi_des=chi_des_att,
            chi_dot_des=chi_dot_des_att,
            Phi=self.Phi_att,
            Lambda=self.Lambda_att,
            K_hat=self.K_hat_att,
            nu=self.nu_att,
            dt=dt,
            yaw_index=2
        )

        # Combine the attitude moment suggested by the unified law
        # with the inner attitude realization.
        #
        # The inner attitude law is dominant because it directly tracks
        # roll_des/pitch_des/yaw_des.
        Mx = tau_att[0]
        My = tau_att[1]
        Mz = tau_att[2]

        Mx = np.clip(Mx, -self.Mx_limit, self.Mx_limit)
        My = np.clip(My, -self.My_limit, self.My_limit)
        Mz = np.clip(Mz, -self.Mz_limit, self.Mz_limit)

        # ======================================================
        # Total thrust
        # ======================================================

        tilt_comp = math.cos(self.roll) * math.cos(self.pitch)
        tilt_comp = max(tilt_comp, 0.5)

        F = self.mass * (self.g + az_cmd) / tilt_comp
        F = max(0.0, F)

        motor_thrusts = self.mixer(
            F,
            Mx,
            My,
            Mz
        )

        motor_msg = Float64MultiArray()
        motor_msg.data = motor_thrusts.tolist()
        self.motor_pub.publish(motor_msg)

        # ======================================================
        # Manipulator torque
        # ======================================================

        if self.have_joint_state and self.q0 is not None:
            tau_arm = np.array([
                tau_alpha[0],
                tau_alpha[1]
            ])

            tau_arm = np.clip(
                tau_arm,
                -self.tau_arm_limit,
                self.tau_arm_limit
            )

            arm_msg = Float64MultiArray()
            arm_msg.data = tau_arm.tolist()
            self.arm_pub.publish(arm_msg)
        else:
            tau_arm = np.zeros(2)

        # ======================================================
        # Logging
        # ======================================================

        dt_log = (now - self.last_log_time).nanoseconds * 1e-9

        if dt_log > 1.0:
            self.last_log_time = now

            self.get_logger().info(
                f"t={t:.1f} | "
                f"pos=[{self.x:.2f}, {self.y:.2f}, {self.z:.3f}] "
                f"pos_des=[{self.x_des:.2f}, {self.y_des:.2f}, {self.z_des:.3f}] "
                f"vel=[{self.x_dot:.2f}, {self.y_dot:.2f}, {self.z_dot:.2f}] | "
                f"rpy=[{self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f}] "
                f"rpy_des=[{roll_des:.3f}, {pitch_des:.3f}, {self.yaw_des:.3f}] | "
                f"e={np.round(e, 3)} "
                f"s={np.round(s, 3)} | "
                f"rho={rho:.3f} "
                f"K_hat={np.round(self.K_hat, 3)} "
                f"xi_norm={xi_norm:.3f} | "
                f"tau_p={np.round(tau_p, 3)} "
                f"tau_q={np.round(tau_q, 3)} "
                f"tau_alpha={np.round(tau_alpha, 3)} | "
                f"att_e={np.round(e_att, 3)} "
                f"att_s={np.round(s_att, 3)} "
                f"rho_att={rho_att:.3f} | "
                f"F={F:.2f} "
                f"M=[{Mx:.2f}, {My:.2f}, {Mz:.2f}] | "
                f"motors={np.round(motor_thrusts, 2)} | "
                f"{arm_phase} "
                f"q={np.round(self.q, 3)} "
                f"q_des={np.round(q_des, 3)} "
                f"e_arm={np.round(e[6:8], 3)} "
                f"s_arm={np.round(s[6:8], 3)} "
                f"tau_arm={np.round(tau_arm, 3)} | "
                f"{drone_phase}"
            )

    def stop_all(self):
        motor_msg = Float64MultiArray()
        motor_msg.data = [
            0.0,
            0.0,
            0.0,
            0.0
        ]
        self.motor_pub.publish(motor_msg)

        arm_msg = Float64MultiArray()
        arm_msg.data = [
            0.0,
            0.0
        ]
        self.arm_pub.publish(arm_msg)


def main(args=None):
    rclpy.init(args=args)

    node = FullAdaptiveController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if rclpy.ok():
                node.stop_all()

        except Exception as e:
            node.get_logger().warn(
                f"Could not send zero commands on shutdown: {e}"
            )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()