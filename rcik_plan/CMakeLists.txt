cmake_minimum_required(VERSION 2.8.3)
project(rcik_plan)

add_definitions(-std=c++11 -O3)
#add_definitions(-std=c++11)

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        moveit_core
        moveit_ros_planning
        moveit_ros_planning_interface
        trac_ik_lib
        torm
        eigen_conversions
        pcl_ros
        )

set(CMAKE_PREFIX_PATH "/home/mincheul/libtorch")
find_package(Torch REQUIRED)
find_package(ompl REQUIRED)
find_package(CUDA REQUIRED)
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME} moveit_collision_distance_field
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OMPL_INCLUDE_DIRS}
        "/home/mincheul/libtorch/include"
)

link_directories("/home/mincheul/libtorch/lib")

add_library(${PROJECT_NAME}
        src/rcik_planning.cpp
        src/rcik_parameters.cpp
        src/rcik_problem.cpp
        )

set(CUDA_NVCC_FLAGS "-arch=compute_75" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${${PROJECT_NAME}_VERSION})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        )

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

add_executable(keyboard src/rcik_keyboard.cpp)
target_link_libraries(keyboard
        ${catkin_LIBRARIES} ${PROJECT_NAME}
        /opt/ros/melodic/lib/libmoveit_collision_distance_field.so
        ${CUDA_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${TORCH_LIBRARIES}
        )
