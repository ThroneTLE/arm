cmake_minimum_required(VERSION 3.0.2)
project(competition_sim_bridge)

find_package(catkin REQUIRED COMPONENTS
  actionlib control_msgs gazebo_msgs geometry_msgs moveit_commander rospy tf2_ros
)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/competition_sim_bridge_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
