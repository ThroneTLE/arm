cmake_minimum_required(VERSION 3.0.2)
project(arm_vision_framework)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs message_filters rospy sensor_msgs std_msgs std_srvs
)

catkin_install_python(PROGRAMS
  scripts/controller_state_node.py
  scripts/pipeline_node.py
  scripts/oak_depthai_node.py
  scripts/run_pipeline.py
  tools/calibration_tool.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
