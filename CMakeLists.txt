cmake_minimum_required (VERSION 3.12)

project ("RobustRegressionLib")
project(RobustRegressionLib VERSION  1.2.2 DESCRIPTION "Library with functions for robust regression and a testapp")

option(WITH_PYTHON "set this to generate python bindings" ON)
###################################################################
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(library)
add_subdirectory(testapp)
