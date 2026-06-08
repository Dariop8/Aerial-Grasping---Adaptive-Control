import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


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
        # Physical parameters
        # ==========================================================

        self.mass = 3.06
        self.g = 9.81

        self.l = 0.25
        self.k_yaw = 0.02

        self.f_min = 0.0
        self.f_max = 15.0

        # NOTA:
        # Non usiamo più payload_mass.
        # Dopo l'attach il controllore continua a usare la massa nominale.
        # Il payload è trattato come incertezza/disturbo, come nello spirito
        # del controllo adattivo del paper.

        # ==========================================================
        # Desired drone position / yaw
        # ==========================================================

        self.x_des = None
        self.y_des = None
        self.z_des = None
        self.yaw_des = None

        self.home_position = None
        self.target_position = None

        # Se il drone spawna a z = 1.5, questo target porta il drone a z = 0.78.
        # Hai già visto che questa quota permette l'attach.
        self.target_offset = np.array([
            1.0,
            0.0,
           -0.74
        ])

        # ==========================================================
        # Grasp / mission handling
        # ==========================================================

        self.object_attached = False
        self.attach_detected_time = None

        # Dopo l'attach facciamo una salita verticale più lenta e meno aggressiva.
        self.lift_after_attach_height = 0.35
        self.lift_after_attach_duration = 8.0

        self.lift_start_position = None
        self.lift_goal_position = None

        # Dopo l'attach, il riferimento del braccio diventa costante.
        # Non è un PD separato: resta dentro la stessa legge adattiva.
        self.q_hold_after_attach = None

        # ==========================================================
        # Mission state machine
        #
        # INITIAL_HOVER
        # -> GO_TO_TARGET
        # -> WAIT_AT_TARGET
        # -> ARM_GRASPING
        # -> LIFT_AFTER_ATTACH   triggered by /grasp/status == attached
        # -> FINAL_HOVER
        # ==========================================================

        self.mission_state = "INITIAL_HOVER"
        self.state_start_time = None

        self.goto_start_position = None
        self.goto_goal_position = None

        self.return_start_position = None
        self.return_goal_position = None

        self.arm_task_start_time = None

        self.initial_hover_time = 5.0
        self.goto_duration = 16.0
        self.stabilization_time = 2.0
        self.return_duration = 16.0

        self.position_tolerance = 0.08
        self.velocity_tolerance = 0.08
        self.attitude_tolerance = deg2rad(5.0)

        # ==========================================================
        # Paper-like adaptive controller, stabilized for Gazebo
        #
        # chi = [x, y, z, roll, pitch, yaw, q1, q2]
        #
        # e = chi - chi_des
        # s = e_dot + Phi e
        # K_hat_dot_i = ||s|| ||xi||^i - nu_i K_hat_i
        # rho = K0 + K1 ||xi|| + K2 ||xi||^2
        # tau = - Lambda s - rho * s / sqrt(||s||^2 + delta)
        # ==========================================================

        self.Phi = np.diag([
            1.0,   # x
            1.0,   # y
            1.2,   # z
            1.1,    # roll
            1.1,    # pitch
            1.0,    # yaw
            1.2,   # joint1
            1.2    # joint2
        ])

        self.Lambda = np.diag([
            2.00,   # x
            2.00,   # y
            3.5,   # z
            1.5,    # roll
            1.5,    # pitch
            1.2,    # yaw
            3.0,   # joint1
            3.0    # joint2
        ])

        self.K_hat = np.array([0.1, 0.1, 0.1], dtype=float)
        self.nu = np.array([2.0, 5.0, 5.0], dtype=float)

        self.delta = 0.1

        # ==========================================================
        # Practical actuator limits
        #
        # Questi limiti non sono il cuore teorico del paper, ma sono
        # necessari in Gazebo per evitare saturazioni numeriche.
        # ==========================================================

        self.ax_limit = 4.0
        self.ay_limit = 4.0
        self.az_limit = 10.0

        self.max_tilt_des = deg2rad(15.0)

        self.Mx_limit = 2.2
        self.My_limit = 2.2
        self.Mz_limit = 1.5

        self.tau_arm_limit = 20.0

        # Protezioni numeriche.
        # Per una simulazione Gazebo stabile conviene tenerle attive.
        self.use_adaptive_caps = True
        self.K_hat_limit = 2.5
        self.rho_limit = 3.0

        # ==========================================================
        # Inner attitude adaptive stabilizer
        #
        # Necessario perché il quadrotore è sottoattuato:
        # tau_x e tau_y non possono essere applicate direttamente
        # come forze laterali. Vengono realizzate tramite roll/pitch.
        # ==========================================================

        self.Phi_att = np.diag([
            4.0,
            4.0,
            2.2
        ])

        self.Lambda_att = np.diag([
            2.8,
            2.8,
            1.2
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

        self.create_subscription(
            String,
            "/grasp/status",
            self.grasp_status_callback,
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

        self.get_logger().info(
            "Paper adaptive controller with attach-lift mission started"
        )
        self.get_logger().info(
            "No explicit payload-mass compensation is used."
        )
        self.get_logger().info(
            "Waiting for /drone/odom, /joint_states and /grasp/status..."
        )

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

            self.home_position = np.array([
                self.x,
                self.y,
                self.z
            ])

            self.target_position = self.home_position + self.target_offset

            self.get_logger().info(
                f"Initial position hold: "
                f"x_des={self.x_des:.3f}, "
                f"y_des={self.y_des:.3f}, "
                f"z_des={self.z_des:.3f}"
            )

            self.get_logger().info(
                f"Mission target: "
                f"x={self.target_position[0]:.3f}, "
                f"y={self.target_position[1]:.3f}, "
                f"z={self.target_position[2]:.3f}"
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

    def grasp_status_callback(self, msg):
        status = msg.data.strip().lower()

        if status == "attached" and not self.object_attached:
            self.object_attached = True
            self.attach_detected_time = self.get_clock().now()

            if self.have_joint_state:
                self.q_hold_after_attach = self.q.copy()
            else:
                self.q_hold_after_attach = np.zeros(2)

            self.get_logger().warn(
                "Object attached detected. Will switch to LIFT_AFTER_ATTACH."
            )

        elif status == "detached":
            if self.object_attached:
                self.get_logger().warn("Object detached detected.")

            self.object_attached = False

    # ==========================================================
    # Mission state machine
    # ==========================================================

    def set_mission_state(self, new_state, t):
        if new_state == self.mission_state:
            return

        old_state = self.mission_state

        self.get_logger().warn(
            f"===== MISSION STATE: {old_state} -> {new_state} at t={t:.2f}s ====="
        )

        self.mission_state = new_state
        self.state_start_time = t

        if new_state == "GO_TO_TARGET":
            self.goto_start_position = np.array([
                self.x,
                self.y,
                self.z
            ])

            self.goto_goal_position = self.target_position.copy()

            self.get_logger().info(
                f"GO_TO_TARGET from {np.round(self.goto_start_position, 3)} "
                f"to {np.round(self.goto_goal_position, 3)}"
            )

        elif new_state == "ARM_GRASPING":
            self.arm_task_start_time = t

            self.get_logger().info(
                f"ARM_GRASPING starts at t={t:.2f}s"
            )

        elif new_state == "LIFT_AFTER_ATTACH":
            self.lift_start_position = np.array([
                self.x,
                self.y,
                self.z
            ])

            self.lift_goal_position = np.array([
                self.x,
                self.y,
                self.z + self.lift_after_attach_height
            ])

            if self.q_hold_after_attach is None:
                self.q_hold_after_attach = self.q.copy()

            self.get_logger().warn(
                f"LIFT_AFTER_ATTACH from {np.round(self.lift_start_position, 3)} "
                f"to {np.round(self.lift_goal_position, 3)}"
            )

        elif new_state == "RETURN_HOME":
            self.return_start_position = np.array([
                self.x,
                self.y,
                self.z
            ])

            self.return_goal_position = self.home_position.copy()

            self.get_logger().info(
                f"RETURN_HOME from {np.round(self.return_start_position, 3)} "
                f"to {np.round(self.return_goal_position, 3)}"
            )

    def position_error_to_target(self):
        if self.target_position is None:
            return 999.0

        current = np.array([
            self.x,
            self.y,
            self.z
        ])

        return np.linalg.norm(current - self.target_position)

    def position_error_to_home(self):
        if self.home_position is None:
            return 999.0

        current = np.array([
            self.x,
            self.y,
            self.z
        ])

        return np.linalg.norm(current - self.home_position)

    def velocity_norm(self):
        return np.linalg.norm(np.array([
            self.x_dot,
            self.y_dot,
            self.z_dot
        ]))

    def is_at_target_and_stable(self):
        return (
            self.position_error_to_target() < self.position_tolerance
            and self.velocity_norm() < self.velocity_tolerance
            and abs(self.roll) < self.attitude_tolerance
            and abs(self.pitch) < self.attitude_tolerance
        )

    def is_at_home_and_stable(self):
        return (
            self.position_error_to_home() < self.position_tolerance
            and self.velocity_norm() < self.velocity_tolerance
            and abs(self.roll) < self.attitude_tolerance
            and abs(self.pitch) < self.attitude_tolerance
        )

    def arm_task_finished(self, arm_local_t):
        return arm_local_t > 73.0

    def update_mission_state(self, t):
        if self.state_start_time is None:
            self.state_start_time = t

        time_in_state = t - self.state_start_time

        # Highest-priority transition:
        # appena l'oggetto viene attaccato, cambia traiettoria desiderata:
        # salita verticale. Non viene modificata la massa nel modello.
        if (
            self.object_attached
            and self.mission_state not in [
                "LIFT_AFTER_ATTACH",
                "FINAL_HOVER"
            ]
        ):
            self.set_mission_state("LIFT_AFTER_ATTACH", t)
            return

        if self.mission_state == "INITIAL_HOVER":
            if time_in_state > self.initial_hover_time:
                self.set_mission_state("GO_TO_TARGET", t)

        elif self.mission_state == "GO_TO_TARGET":
            if time_in_state > self.goto_duration and self.is_at_target_and_stable():
                self.set_mission_state("WAIT_AT_TARGET", t)

        elif self.mission_state == "WAIT_AT_TARGET":
            if self.have_joint_state and self.q0 is not None:
                if time_in_state > self.stabilization_time:
                    self.set_mission_state("ARM_GRASPING", t)

        elif self.mission_state == "ARM_GRASPING":
            if self.arm_task_start_time is not None:
                arm_local_t = t - self.arm_task_start_time

                if self.arm_task_finished(arm_local_t):
                    self.set_mission_state("RETURN_HOME", t)

        elif self.mission_state == "LIFT_AFTER_ATTACH":
            if time_in_state > self.lift_after_attach_duration + 1.0:
                self.set_mission_state("FINAL_HOVER", t)

        elif self.mission_state == "RETURN_HOME":
            if time_in_state > self.return_duration and self.is_at_home_and_stable():
                self.set_mission_state("FINAL_HOVER", t)

        elif self.mission_state == "FINAL_HOVER":
            pass

    # ==========================================================
    # Trajectory helpers
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

    def interpolate_position(self, p_start, p_goal, t, t_start, duration):
        if t <= t_start:
            return p_start, np.zeros(3)

        if t >= t_start + duration:
            return p_goal, np.zeros(3)

        s = (t - t_start) / duration

        sigma = smoothstep(s)
        sigma_dot = smoothstep_dot(s) / duration

        p_des = p_start + sigma * (p_goal - p_start)
        p_dot_des = sigma_dot * (p_goal - p_start)

        return p_des, p_dot_des

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

    # ==========================================================
    # Desired drone trajectory from mission state
    # ==========================================================

    def desired_drone_trajectory(self, t):
        yaw0 = self.yaw_des

        if self.mission_state == "INITIAL_HOVER":
            return (
                np.array([
                    self.home_position[0],
                    self.home_position[1],
                    self.home_position[2],
                    yaw0
                ]),
                np.zeros(4),
                "DRONE_INITIAL_HOVER"
            )

        elif self.mission_state == "GO_TO_TARGET":
            local_t = t - self.state_start_time

            x_des, x_dot_des = self.interpolate_scalar(
                self.goto_start_position[0],
                self.goto_goal_position[0],
                local_t,
                t_start=0.0,
                duration=self.goto_duration
            )

            y_des, y_dot_des = self.interpolate_scalar(
                self.goto_start_position[1],
                self.goto_goal_position[1],
                local_t,
                t_start=0.0,
                duration=self.goto_duration
            )

            z_des, z_dot_des = self.interpolate_scalar(
                self.goto_start_position[2],
                self.goto_goal_position[2],
                local_t,
                t_start=0.0,
                duration=self.goto_duration
            )

            return (
                np.array([
                    x_des,
                    y_des,
                    z_des,
                    yaw0
                ]),
                np.array([
                    x_dot_des,
                    y_dot_des,
                    z_dot_des,
                    0.0
                ]),
                "DRONE_GO_TO_TARGET"
            )

        elif self.mission_state == "WAIT_AT_TARGET":
            return (
                np.array([
                    self.target_position[0],
                    self.target_position[1],
                    self.target_position[2],
                    yaw0
                ]),
                np.zeros(4),
                "DRONE_WAIT_AT_TARGET"
            )

        elif self.mission_state == "ARM_GRASPING":
            return (
                np.array([
                    self.target_position[0],
                    self.target_position[1],
                    self.target_position[2],
                    yaw0
                ]),
                np.zeros(4),
                "DRONE_HOLD_TARGET_DURING_GRASP"
            )

        elif self.mission_state == "LIFT_AFTER_ATTACH":
            if self.lift_start_position is None or self.lift_goal_position is None:
                return (
                    np.array([
                        self.x,
                        self.y,
                        self.z,
                        yaw0
                    ]),
                    np.zeros(4),
                    "DRONE_LIFT_AFTER_ATTACH_INIT"
                )

            local_t = t - self.state_start_time

            pos_des, pos_dot_des = self.interpolate_position(
                self.lift_start_position,
                self.lift_goal_position,
                local_t,
                t_start=0.0,
                duration=self.lift_after_attach_duration
            )

            return (
                np.array([
                    pos_des[0],
                    pos_des[1],
                    pos_des[2],
                    yaw0
                ]),
                np.array([
                    pos_dot_des[0],
                    pos_dot_des[1],
                    pos_dot_des[2],
                    0.0
                ]),
                "DRONE_LIFT_AFTER_ATTACH"
            )

        elif self.mission_state == "RETURN_HOME":
            local_t = t - self.state_start_time

            x_des, x_dot_des = self.interpolate_scalar(
                self.return_start_position[0],
                self.return_goal_position[0],
                local_t,
                t_start=0.0,
                duration=self.return_duration
            )

            y_des, y_dot_des = self.interpolate_scalar(
                self.return_start_position[1],
                self.return_goal_position[1],
                local_t,
                t_start=0.0,
                duration=self.return_duration
            )

            z_des, z_dot_des = self.interpolate_scalar(
                self.return_start_position[2],
                self.return_goal_position[2],
                local_t,
                t_start=0.0,
                duration=self.return_duration
            )

            return (
                np.array([
                    x_des,
                    y_des,
                    z_des,
                    yaw0
                ]),
                np.array([
                    x_dot_des,
                    y_dot_des,
                    z_dot_des,
                    0.0
                ]),
                "DRONE_RETURN_HOME"
            )

        else:
            # FINAL_HOVER.
            # Dopo l'attach e il lift resta in hover alla quota finale.
            if self.lift_goal_position is not None:
                hover_pos = self.lift_goal_position
            else:
                hover_pos = self.home_position

            return (
                np.array([
                    hover_pos[0],
                    hover_pos[1],
                    hover_pos[2],
                    yaw0
                ]),
                np.zeros(4),
                "DRONE_FINAL_HOVER"
            )

    # ==========================================================
    # Desired manipulator trajectory from mission state
    # ==========================================================

    def desired_arm_trajectory(self, t):
        if self.q0 is None:
            return (
                self.q.copy(),
                np.zeros(2),
                "ARM_WAITING_FOR_JOINT_STATES"
            )

        # Dopo attach, la traiettoria desiderata del braccio diventa costante.
        # Questo non introduce un controllore esterno: q_des resta dentro chi_des
        # e viene inseguito dalla stessa legge adattiva.
        if self.object_attached and self.q_hold_after_attach is not None:
            return (
                self.q_hold_after_attach.copy(),
                np.zeros(2),
                "ARM_HOLD_AFTER_ATTACH"
            )

        q_home = self.q0.copy()

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

        if self.mission_state != "ARM_GRASPING":
            return (
                q_home,
                np.zeros(2),
                "ARM_WAIT_FOR_DRONE_TARGET"
            )

        arm_t = t - self.arm_task_start_time

        if arm_t < 6.0:
            return (
                q_home,
                np.zeros(2),
                "ARM_INITIAL_HOLD"
            )

        elif arm_t < 22.0:
            q_des, q_dot_des = self.interpolate(
                q_home,
                q_pre_grasp,
                arm_t,
                t_start=6.0,
                duration=16.0
            )

            return (
                q_des,
                q_dot_des,
                "ARM_MOVE_TO_PRE_GRASP"
            )

        elif arm_t < 36.0:
            q_des, q_dot_des = self.interpolate(
                q_pre_grasp,
                q_grasp,
                arm_t,
                t_start=22.0,
                duration=14.0
            )

            return (
                q_des,
                q_dot_des,
                "ARM_MOVE_TO_GRASP"
            )

        elif arm_t < 43.0:
            return (
                q_grasp,
                np.zeros(2),
                "ARM_GRASP_HOLD"
            )

        elif arm_t < 57.0:
            q_des, q_dot_des = self.interpolate(
                q_grasp,
                q_retract,
                arm_t,
                t_start=43.0,
                duration=14.0
            )

            return (
                q_des,
                q_dot_des,
                "ARM_RETRACT"
            )

        elif arm_t < 73.0:
            q_des, q_dot_des = self.interpolate(
                q_retract,
                q_home,
                arm_t,
                t_start=57.0,
                duration=16.0
            )

            return (
                q_des,
                q_dot_des,
                "ARM_RETURN_HOME"
            )

        else:
            return (
                q_home,
                np.zeros(2),
                "ARM_HOME_HOLD_FINAL"
            )

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
        # e = chi - chi_des
        e = chi - chi_des

        if yaw_index is not None:
            e[yaw_index] = wrap_angle(e[yaw_index])

        # e_dot = chi_dot - chi_dot_des
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

        # Protezione numerica per Gazebo.
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

        # Smooth version of s / ||s||
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
            or self.home_position is None
            or self.target_position is None
        ):
            return

        now = self.get_clock().now()

        if self.start_time is None:
            self.start_time = now
            self.last_time = now
            self.last_log_time = now
            self.state_start_time = 0.0

            self.get_logger().info("Controller time initialized")

            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.1:
            self.get_logger().warn(f"Invalid dt={dt:.4f}, skipping")
            return

        t = (now - self.start_time).nanoseconds * 1e-9

        self.update_mission_state(t)

        # ======================================================
        # Desired trajectories
        # ======================================================

        drone_des, drone_dot_des, drone_phase = self.desired_drone_trajectory(t)

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

        chi_des = np.array([
            drone_des[0],
            drone_des[1],
            drone_des[2],
            0.0,
            0.0,
            drone_des[3],
            q_des[0],
            q_des[1]
        ])

        chi_dot_des = np.array([
            drone_dot_des[0],
            drone_dot_des[1],
            drone_dot_des[2],
            0.0,
            0.0,
            drone_dot_des[3],
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

        tau_p = tau[0:3]
        tau_q = tau[3:6]
        tau_alpha = tau[6:8]

        # ======================================================
        # Position part
        #
        # tau_x e tau_y sono trasformati in roll/pitch desiderati
        # perché il quadrotore è sottoattuato.
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
            drone_des[3]
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

        Mx = np.clip(tau_att[0], -self.Mx_limit, self.Mx_limit)
        My = np.clip(tau_att[1], -self.My_limit, self.My_limit)
        Mz = np.clip(tau_att[2], -self.Mz_limit, self.Mz_limit)

        # ======================================================
        # Total thrust
        #
        # Importante:
        # qui usiamo SOLO la massa nominale del drone.
        # Nessuna compensazione esplicita del payload.
        # L'oggetto attaccato è trattato come disturbo/incertezza.
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
                f"mission={self.mission_state} | "
                f"{drone_phase} | "
                f"attached={self.object_attached} | "
                f"pos=[{self.x:.2f}, {self.y:.2f}, {self.z:.3f}] "
                f"pos_des=[{drone_des[0]:.2f}, {drone_des[1]:.2f}, {drone_des[2]:.3f}] "
                f"target_err={self.position_error_to_target():.3f} "
                f"home_err={self.position_error_to_home():.3f} "
                f"vel_norm={self.velocity_norm():.3f} | "
                f"vel=[{self.x_dot:.2f}, {self.y_dot:.2f}, {self.z_dot:.2f}] | "
                f"rpy=[{self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f}] "
                f"rpy_des=[{roll_des:.3f}, {pitch_des:.3f}, {drone_des[3]:.3f}] | "
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
                f"mass_model={self.mass:.2f} "
                f"F={F:.2f} "
                f"M=[{Mx:.2f}, {My:.2f}, {Mz:.2f}] | "
                f"motors={np.round(motor_thrusts, 2)} | "
                f"{arm_phase} "
                f"q={np.round(self.q, 3)} "
                f"q_des={np.round(q_des, 3)} "
                f"e_arm={np.round(e[6:8], 3)} "
                f"s_arm={np.round(s[6:8], 3)} "
                f"tau_arm={np.round(tau_arm, 3)}"
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