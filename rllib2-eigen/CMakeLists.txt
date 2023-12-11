cmake_minimum_required(VERSION 3.10)

project(rllib2-eigen) 

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(PACKAGE_VERSION "1.00")
SET(PACKAGE_LICENSE "GPL")
SET(PROJECT_DESCRIPTION_SUMMARY "Extends rllib2 with algorithms based on Eigen")
SET(CMAKE_BUILD_TYPE Release)
SET(PROJECT_CFLAGS "-std=c++20")

SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/" CACHE PATH "The subdirectory to the header prefix")

find_package(PkgConfig REQUIRED)
pkg_check_modules(RLLIB2 rllib2 REQUIRED)
set(PKG_CONFIG_DEPENDS "rllib2")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
message("Eigen is in ${EIGEN3_INCLUDE_DIR}")

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
"
Name: ${CMAKE_PROJECT_NAME}
Description: ${PROJECT_DESCRIPTION_SUMMARY}
Version: ${PACKAGE_VERSION}
Requires: rllib2 eigen3
Libs: 
Cflags: -I${INCLUDE_INSTALL_DIR}
"
)

# Define the path to install the pc pkg-config file
SET(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/" CACHE STRING "Base directory for pkgconfig files")

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
    	DESTINATION ${PKGCONFIG_INSTALL_DIR}
	COMPONENT devel)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(doc)


