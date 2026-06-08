#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <memory>
#include <thread>
#include <mutex>
#include <array>

namespace gazebo
{

class MotorThrustPlugin : public ModelPlugin
{
public:
  MotorThrustPlugin()
  {
    thrusts_.fill(0.0);
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();

    if (sdf->HasElement("link_name")) {
      link_name_ = sdf->Get<std::string>("link_name");
    } else {
      link_name_ = "base_link";
    }

    base_link_ = model_->GetLink(link_name_);

    if (!base_link_) {
      gzerr << "[MotorThrustPlugin] Link not found: " << link_name_ << std::endl;
      return;
    }

    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    node_ = std::make_shared<rclcpp::Node>("motor_thrust_plugin");

    sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/motor_thrust_commands",
      10,
      std::bind(&MotorThrustPlugin::thrustCallback, this, std::placeholders::_1)
    );

    // Rotor positions relative to base_link frame.
    // Must match the URDF rotor positions.
    rotor_positions_[0] = ignition::math::Vector3d( 0.25,  0.25, 0.05);
    rotor_positions_[1] = ignition::math::Vector3d(-0.25,  0.25, 0.05);
    rotor_positions_[2] = ignition::math::Vector3d(-0.25, -0.25, 0.05);
    rotor_positions_[3] = ignition::math::Vector3d( 0.25, -0.25, 0.05);

    // Rotor yaw directions: +1 / -1 for reaction torques.
    rotor_directions_[0] =  1.0;
    rotor_directions_[1] = -1.0;
    rotor_directions_[2] =  1.0;
    rotor_directions_[3] = -1.0;

    // Yaw moment coefficient.
    // This is a simplified value, to be tuned later.
    moment_coefficient_ = 0.02;

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&MotorThrustPlugin::onUpdate, this)
    );

    spin_thread_ = std::thread([this]() {
      rclcpp::spin(node_);
    });

    gzmsg << "[MotorThrustPlugin] Loaded on link: " << link_name_ << std::endl;
  }

  void thrustCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 4) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    for (int i = 0; i < 4; ++i) {
      // Thrust cannot be negative.
      thrusts_[i] = std::max(0.0, msg->data[i]);
    }
  }

  void onUpdate()
  {
    if (!base_link_) {
      return;
    }

    std::array<double, 4> thrusts;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      thrusts = thrusts_;
    }

    ignition::math::Pose3d base_pose = base_link_->WorldPose();
    ignition::math::Quaterniond base_rot = base_pose.Rot();

    for (int i = 0; i < 4; ++i) {
      double f = thrusts[i];

      // Force along local +Z of base_link.
      ignition::math::Vector3d local_force(0.0, 0.0, f);
      ignition::math::Vector3d world_force = base_rot.RotateVector(local_force);

      // Rotor application point in world coordinates.
      ignition::math::Vector3d world_pos =
        base_pose.Pos() + base_rot.RotateVector(rotor_positions_[i]);

      base_link_->AddForceAtWorldPosition(world_force, world_pos);

      // Simplified yaw reaction torque around local Z.
      double yaw_torque = rotor_directions_[i] * moment_coefficient_ * f;
      ignition::math::Vector3d local_torque(0.0, 0.0, yaw_torque);
      ignition::math::Vector3d world_torque = base_rot.RotateVector(local_torque);

      base_link_->AddTorque(world_torque);
    }
  }

  ~MotorThrustPlugin()
	{
	  if (node_) {
	    rclcpp::shutdown();
	  }

	  if (spin_thread_.joinable()) {
	    spin_thread_.join();
	  }
	}
	
private:
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  physics::LinkPtr base_link_;

  event::ConnectionPtr update_connection_;

  std::string link_name_;

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;

  std::thread spin_thread_;
  std::mutex mutex_;

  std::array<double, 4> thrusts_;
  std::array<ignition::math::Vector3d, 4> rotor_positions_;
  std::array<double, 4> rotor_directions_;

  double moment_coefficient_;
};

GZ_REGISTER_MODEL_PLUGIN(MotorThrustPlugin)

}
