cmake_minimum_required (VERSION 3.5)

project ("RobustRegressionLib")
project(RobustRegressionLib VERSION 1.3.2 DESCRIPTION "Library with functions for robust regression and a testapp")

option(WITH_PYTHON "set this to generate python bindings" ON)
option(WITH_TESTAPP "set this to generate the testapplications" ON)
###################################################################
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputfolder)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputfolder)

add_subdirectory(library)

if(${WITH_TESTAPP})
	add_subdirectory(testapp)
endif()
