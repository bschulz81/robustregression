
include(GenerateExportHeader)

set(sources  ${CMAKE_CURRENT_SOURCE_DIR}/src/linearregression.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/src/matrixcode.cpp 
             ${CMAKE_CURRENT_SOURCE_DIR}/src/robustregression.cpp 
             ${CMAKE_CURRENT_SOURCE_DIR}/src/statisticfunctions.cpp 
             ${CMAKE_CURRENT_SOURCE_DIR}/src/nonlinearregression.cpp)

ADD_LIBRARY(RobustRegressionLib SHARED ${sources} "include/lossfunctions.h")

target_include_directories(RobustRegressionLib PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${CMAKE_CURRENT_BINARY_DIR})

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
    add_definitions(-DMACOSX)
    set_property(TARGET RobustRegressionLib PROPERTY CXX_STANDARD 14)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   set(CLANGCOMPILER TRUE)
   add_definitions(-DCLANGCOMPILER)
   set_property(TARGET RobustRegressionLib PROPERTY CXX_STANDARD 14)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   set(GNUCOMPILER TRUE)
   add_definitions(-DGNUCOMPILER)
   set_property(TARGET RobustRegressionLib PROPERTY CXX_STANDARD 14)
endif()



if(MSVC)
    set_property(TARGET RobustRegressionLib PROPERTY CXX_STANDARD 17)
    set_property(TARGET RobustRegressionLib PROPERTY CXX_STANDARD_REQUIRED ON)
    target_compile_options(RobustRegressionLib PUBLIC "/Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} /std:c++17 /EHsc")
endif()


GENERATE_EXPORT_HEADER(RobustRegressionLib           
    BASE_NAME RobustRegressionLib
    EXPORT_MACRO_NAME ROBUSTREGRESSION_API
    EXPORT_FILE_NAME robustregressionlib_exports.h
    STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)


if(${WITH_PYTHON})

    find_package(Python3 COMPONENTS Interpreter Development)  
    find_package(pybind11 CONFIG REQUIRED )

    set_target_properties(Python3::Module PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG ";RELEASE")

    pybind11_add_module(pyRobustRegressionLib ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/pythonbindings.cpp)

    target_include_directories(pyRobustRegressionLib PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${CMAKE_CURRENT_BINARY_DIR})


    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
         set(MACOSX TRUE)
        add_definitions(-DMACOSX)
        set_property(TARGET pyRobustRegressionLib PROPERTY CXX_STANDARD 14)
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CLANGCOMPILER TRUE)
        add_definitions(-DCLANGCOMPILER)
         set_property(TARGET pyRobustRegressionLib PROPERTY CXX_STANDARD 14)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(GNUCOMPILER TRUE)
        add_definitions(-DGNUCOMPILER)
         set_property(TARGET pyRobustRegressionLib PROPERTY CXX_STANDARD 14)
    endif()


    if(MSVC)
        set_property(TARGET pyRobustRegressionLib PROPERTY CXX_STANDARD 17)
        set_property(TARGET pyRobustRegressionLib PROPERTY CXX_STANDARD_REQUIRED ON)
        target_compile_options(pyRobustRegressionLib PUBLIC "/Zc:__cplusplus")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} /std:c++17 /EHsc")
    endif()

    target_link_libraries(pyRobustRegressionLib PUBLIC RobustRegressionLib)

endif()
