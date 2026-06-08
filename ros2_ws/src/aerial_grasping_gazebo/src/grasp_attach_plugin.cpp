#include <memory>
#include <string>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Events.hh>
#include <gazebo_ros/node.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace gazebo
{

class GraspAttachPlugin : public ModelPlugin
{
public:
  GraspAttachPlugin() = default;

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();

    node_ = gazebo_ros::Node::Get(sdf);

    if (sdf->HasElement("gripper_link")) {
      gripper_link_name_ = sdf->Get<std::string>("gripper_link");
    }

    if (sdf->HasElement("object_model")) {
      object_model_name_ = sdf->Get<std::string>("object_model");
    }

    if (sdf->HasElement("object_link")) {
      object_link_name_ = sdf->Get<std::string>("object_link");
    }

    if (sdf->HasElement("attach_distance")) {
      attach_distance_ = sdf->Get<double>("attach_distance");
    }

    if (sdf->HasElement("auto_attach")) {
      auto_attach_ = sdf->Get<bool>("auto_attach");
    }

    if (sdf->HasElement("detach_on_command")) {
      detach_on_command_ = sdf->Get<bool>("detach_on_command");
    }

    gripper_link_ = model_->GetLink(gripper_link_name_);

    if (!gripper_link_) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "GraspAttachPlugin: gripper link [%s] not found in model [%s]",
        gripper_link_name_.c_str(),
        model_->GetName().c_str()
      );
      return;
    }

    status_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/grasp/status",
      10
    );

    command_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "/grasp/command",
      10,
      std::bind(&GraspAttachPlugin::commandCallback, this, std::placeholders::_1)
    );

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&GraspAttachPlugin::onUpdate, this)
    );

    RCLCPP_INFO(
      node_->get_logger(),
      "GraspAttachPlugin loaded. gripper_link=%s object_model=%s object_link=%s attach_distance=%.3f auto_attach=%s",
      gripper_link_name_.c_str(),
      object_model_name_.c_str(),
      object_link_name_.c_str(),
      attach_distance_,
      auto_attach_ ? "true" : "false"
    );

    publishStatus("detached");
  }

private:
  void commandCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string cmd = msg->data;

    if (cmd == "detach" && detach_on_command_) {
      detach();
      return;
    }

    if (cmd == "enable_auto_attach") {
      auto_attach_ = true;
      publishStatus("auto_attach_enabled");
      return;
    }

    if (cmd == "disable_auto_attach") {
      auto_attach_ = false;
      publishStatus("auto_attach_disabled");
      return;
    }

    if (cmd == "status") {
      publishStatus(attached_ ? "attached" : "detached");
      return;
    }

    RCLCPP_WARN(
      node_->get_logger(),
      "Unknown grasp command: %s",
      cmd.c_str()
    );
  }

  void onUpdate()
  {
    if (!auto_attach_) {
      return;
    }

    if (attached_) {
      return;
    }

    if (!updateObjectPointers()) {
      return;
    }

    const double distance = gripperObjectDistance();

    publishDistanceThrottled(distance);

    if (distance < attach_distance_) {
      attach(distance);
    }
  }

  bool updateObjectPointers()
  {
    if (!object_model_) {
      object_model_ = world_->ModelByName(object_model_name_);
    }

    if (!object_model_) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        2000,
        "Object model [%s] not found",
        object_model_name_.c_str()
      );
      return false;
    }

    if (!object_link_) {
      object_link_ = object_model_->GetLink(object_link_name_);
    }

    if (!object_link_) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        2000,
        "Object link [%s] not found",
        object_link_name_.c_str()
      );
      return false;
    }

    return true;
  }

  double gripperObjectDistance()
  {
    const auto pg = gripper_link_->WorldPose().Pos();
    const auto po = object_link_->WorldPose().Pos();

    return pg.Distance(po);
  }

  void attach(double distance)
  {
    if (attached_) {
      return;
    }

    if (!gripper_link_ || !object_link_) {
      publishStatus("attach_failed_missing_link");
      return;
    }

    fixed_joint_ = world_->Physics()->CreateJoint("fixed", model_);

    if (!fixed_joint_) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create fixed joint");
      publishStatus("attach_failed_create_joint");
      return;
    }

    fixed_joint_->Attach(gripper_link_, object_link_);
    fixed_joint_->Load(gripper_link_, object_link_, ignition::math::Pose3d());
    fixed_joint_->Init();

    // Da questo momento l'oggetto diventa payload reale.
    object_model_->SetGravityMode(true);
    object_link_->SetGravityMode(true);

    attached_ = true;

    RCLCPP_INFO(
      node_->get_logger(),
      "AUTO ATTACH: object attached to gripper. distance=%.4f",
      distance
    );

    publishStatus("attached");
  }

  void detach()
  {
    if (!attached_) {
      publishStatus("already_detached");
      return;
    }

    if (fixed_joint_) {
      fixed_joint_->Detach();
      fixed_joint_.reset();
    }

    attached_ = false;

    RCLCPP_INFO(node_->get_logger(), "Object detached");
    publishStatus("detached");
  }

  void publishStatus(const std::string & status_text)
  {
    std_msgs::msg::String msg;
    msg.data = status_text;
    status_pub_->publish(msg);
  }

  void publishDistanceThrottled(double distance)
  {
    RCLCPP_INFO_THROTTLE(
      node_->get_logger(),
      *node_->get_clock(),
      1000,
      "gripper-object distance = %.4f, attached = %s",
      distance,
      attached_ ? "true" : "false"
    );
  }

private:
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  physics::LinkPtr gripper_link_;
  physics::ModelPtr object_model_;
  physics::LinkPtr object_link_;
  physics::JointPtr fixed_joint_;

  gazebo_ros::Node::SharedPtr node_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;

  event::ConnectionPtr update_connection_;

  std::string gripper_link_name_ = "gripper";
  std::string object_model_name_ = "grasp_object";
  std::string object_link_name_ = "object_link";

  double attach_distance_ = 0.08;

  bool auto_attach_ = true;
  bool detach_on_command_ = true;
  bool attached_ = false;
};

GZ_REGISTER_MODEL_PLUGIN(GraspAttachPlugin)

}  // namespace gazebo