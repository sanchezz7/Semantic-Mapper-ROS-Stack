cmake_minimum_required(VERSION 2.8.3)
project(object_detection)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  std_msgs
  sensor_msgs
)

catkin_python_setup()

add_message_files(
   FILES
   Object.msg
   Detections.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_detection
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS bin/object_detection_tf 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/object_detection_dk 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

