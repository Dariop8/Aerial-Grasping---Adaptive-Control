cmake_minimum_required(VERSION 3.8)
project(aerial_grasping_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)

find_package(gazebo REQUIRED)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

add_library(motor_thrust_plugin SHARED
  src/motor_thrust_plugin.cpp
)

ament_target_dependencies(motor_thrust_plugin
  rclcpp
  std_msgs
  gazebo_ros
)

target_link_libraries(motor_thrust_plugin
  ${GAZEBO_LIBRARIES}
)

install(
  TARGETS motor_thrust_plugin
  LIBRARY DESTINATION lib
)

add_library(grasp_attach_plugin SHARED
  src/grasp_attach_plugin.cpp
)

ament_target_dependencies(grasp_attach_plugin
  rclcpp
  std_msgs
  gazebo_ros
)

target_include_directories(grasp_attach_plugin PUBLIC
  ${GAZEBO_INCLUDE_DIRS}
)

target_link_libraries(grasp_attach_plugin
  ${GAZEBO_LIBRARIES}
)

install(TARGETS
  grasp_attach_plugin
  DESTINATION lib
)

ament_package()
