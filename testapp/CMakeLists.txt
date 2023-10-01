link_directories(${CMAKE_BINARY_DIR}/outputfolder)
include_directories(${CMAKE_BINARY_DIR}/library/include)
include_directories(${CMAKE_BINARY_DIR}/library)
add_executable(testapplication "testapplication.cpp")
target_link_libraries(testapplication PUBLIC RobustRegressionLib)

if (${WITH_PYTHON})
    add_custom_command(
                    TARGET testapplication POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy 
                     ${CMAKE_CURRENT_SOURCE_DIR}/testapp.py 
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)

endif()

