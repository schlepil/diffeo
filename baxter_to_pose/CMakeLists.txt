cmake_minimum_required(VERSION 2.8.3)
project(baxter_to_pose)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fPIC")




### adding sources ###
set(HEADERS
	cppDiffeo/diffeoClean.hpp
	cppDiffeo/diffeoMethods.hpp
	cppDiffeo/diffeoMovements.hpp
	cppDiffeo/diffeoSearch.hpp
	cppDiffeo/FileVector.h
	cppDiffeo/parameterisations.hpp
	cppDiffeo/thingsThatShouldExist.hpp
)

include_directories(
  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} 
)

catkin_package(
)

add_executable(gripper_control src/gripper_control.cpp)
target_link_libraries(gripper_control ${catkin_LIBRARIES})

add_executable(real_baxter_arm_diffeomorphism src/real_baxter_arm_diffeomorphism.cpp ${HEADERS})
target_link_libraries(real_baxter_arm_diffeomorphism ${catkin_LIBRARIES})

add_executable(arm_backdrive src/arm_backdrive.cpp ${HEADERS})
target_link_libraries(arm_backdrive ${catkin_LIBRARIES})

add_executable(arm_controller_to_target_pose src/arm_controller_to_target_pose.cpp)
target_link_libraries(arm_controller_to_target_pose ${catkin_LIBRARIES})

add_executable(second_move src/second_move.cpp)
target_link_libraries(second_move ${catkin_LIBRARIES})
