cmake_minimum_required (VERSION 2.8.11)

project (diffeo)

set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 11)



SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -L/usr/lib/x86_64-linux-gnu -ltinyxml" )

### adding Eigen3 ###
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
find_package(tinyXML REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)


### adding sources ###
set(HEADERS
	diffeoClean.hpp
	diffeoMethods.hpp
	diffeoMovements.hpp
	diffeoSearch.hpp
	FileVector.h
	parameterisations.hpp
	thingsThatShouldExist.hpp
)


add_executable(diffeoMethods main.cpp ${HEADERS})

install (TARGETS diffeoMethods RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../python/bin/Release)
