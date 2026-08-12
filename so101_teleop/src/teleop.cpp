#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

class FollowerCommandRelay : public rclcpp::Node
{
public:
  FollowerCommandRelay()
  : Node("leader_follower_teleop")
  {
    RCLCPP_INFO(get_logger(), "Initializing FollowerCommandRelay...");

    leader_topic_ = declare_parameter<std::string>("leader_topic", "/leader/joint_states");
    follower_fwd_topic_ =
      declare_parameter<std::string>("fwd_topic", "/follower/forward_controller/commands");

    stale_timeout_s_ = declare_parameter<double>("stale_timeout_s", 0.25);

    arm_joints_ = declare_parameter<std::vector<std::string>>(
        "arm_joints", std::vector<std::string>{"shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper"});

    RCLCPP_INFO(get_logger(), "Leader: %s", leader_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Follower forward commands: %s", follower_fwd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Rate: %.1f Hz, Arm joints: %zu", kPublishRateHz,
                arm_joints_.size());

    // ROS interfaces
    leader_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        leader_topic_, rclcpp::SensorDataQoS(),
        std::bind(&FollowerCommandRelay::joint_state_callback, this, std::placeholders::_1));

    forward_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(follower_fwd_topic_,
                                                                      rclcpp::QoS(10).reliable());

    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / kPublishRateHz),
                               std::bind(&FollowerCommandRelay::control_loop, this));

    raw_arm_.resize(arm_joints_.size(), 0.0);

    RCLCPP_INFO(get_logger(), "FollowerCommandRelay initialized.");
  }

private:
  // Parameters
  static constexpr double kPublishRateHz = 30.0;
  std::string leader_topic_;
  std::string follower_fwd_topic_;
  double stale_timeout_s_{0.25};
  std::vector<std::string> arm_joints_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr leader_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr forward_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  bool initialized_{false};
  std::vector<int> arm_idx_;
  std::vector<double> raw_arm_;
  rclcpp::Time last_leader_stamp_{0, 0, RCL_ROS_TIME};

  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {

    if (!initialized_ && !initialize_indices(*msg)) {return;}

    last_leader_stamp_ = this->now();

    // Cache targets
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      raw_arm_[i] = msg->position[arm_idx_[i]];
    }
  }

  bool initialize_indices(const sensor_msgs::msg::JointState & msg)
  {
    // Build name -> index map
    std::unordered_map<std::string, int> idx_map;
    idx_map.reserve(msg.name.size());
    for (size_t i = 0; i < msg.name.size(); i++) {
      idx_map[msg.name[i]] = static_cast<int>(i);
    }

    // Arm indexes
    arm_idx_.assign(arm_joints_.size(), -1);
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      auto it = idx_map.find(arm_joints_[i]);
      if (it == idx_map.end()) {
        RCLCPP_ERROR(this->get_logger(), "Leader arm joint '%s' not found", arm_joints_[i].c_str());
        return false;
      }
      arm_idx_[i] = it->second;
    }

    initialized_ = true;
    RCLCPP_INFO(get_logger(), "Initialized: %zu arm joints", arm_joints_.size());
    return true;
  }

  void control_loop()
  {

    if (!initialized_) {return;}

    const auto now = this->now();

    // Leader data stale: do nothing (holds last command on follower)
    if ((now - last_leader_stamp_).seconds() > stale_timeout_s_) {return;}

    publish_arm();
  }

  void publish_arm()
  {
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = raw_arm_;
    forward_pub_->publish(cmd);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FollowerCommandRelay>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
