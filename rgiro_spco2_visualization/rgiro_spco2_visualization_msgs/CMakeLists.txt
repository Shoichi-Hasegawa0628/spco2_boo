cmake_minimum_required(VERSION 2.8.3)
project(rgiro_spco2_visualization_msgs)

add_definitions(-std=c++14)

find_package(catkin REQUIRED
  COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  GaussianDistribution.msg
  GaussianDistributions.msg
)

add_service_files(
  FILES
  GaussianService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  rospy
  std_msgs
  message_runtime
)
