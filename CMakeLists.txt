cmake_minimum_required(VERSION 2.8.3)
project(cup_demo)

find_package(
 catkin REQUIRED COMPONENTS
 roscpp
 sensor_msgs
 std_msgs
 message_generation
 rostime message_filters cv_bridge image_transport compressed_image_transport compressed_depth_image_transport kinect2_bridge
)

find_package(
 PCL 1.7 REQUIRED PATHS /usr NO_DEFAULT_PATH
)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

include_directories(
 ${EIGEN_INCLUDE_DIRS}
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_definitions(${EIGEN_DEFINITIONS})

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


catkin_package(

)


execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME}
  SOURCES
  ${FILES_LIST}
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

##Excutables
add_executable(cup_demo src/cup_demo.cpp)
target_link_libraries(cup_demo ${catkin_LIBRARIES} ${PCL_LIBRARIES})


