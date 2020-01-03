cmake_minimum_required(VERSION 2.8.3)
project(obstacle_detector_pca)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    bounding_box_lib
)

find_package(PCL 1.8 REQUIRED)

find_package(Eigen3 REQUIRED COMPONENTS system)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_detector_pca
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(obstacle_detector_pca src/obstacle_detector_pca.cpp)

add_dependencies(obstacle_detector_pca ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(obstacle_detector_pca
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
