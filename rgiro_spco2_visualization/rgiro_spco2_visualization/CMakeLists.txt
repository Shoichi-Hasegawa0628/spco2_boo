cmake_minimum_required(VERSION 2.8.3)
project(rgiro_spco2_visualization)

add_definitions(-std=c++14)

find_package(catkin REQUIRED
  COMPONENTS
  rospy
  roscpp
  std_msgs
  visualization_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  rgiro_spco2_visualization_msgs
)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  rospy
  roscpp
  std_msgs
  visualization_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  rgiro_spco2_visualization_msgs
)

add_library(${PROJECT_NAME} src/gaussian_distribution.cpp src/modules/animation.cpp)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_msgs_generate_messages_cpp)

add_executable(gaussian_distribution_node src/gaussian_distribution_node.cpp)

target_link_libraries(gaussian_distribution_node ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
