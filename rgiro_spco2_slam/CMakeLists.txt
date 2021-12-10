cmake_minimum_required(VERSION 2.8.3)
project(rgiro_spco2_slam)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(
  FILES
  spco_data_image.srv
 )

 generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
