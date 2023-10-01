
include(GenerateExportHeader)

set (librarydir ${CMAKE_CURRENT_SOURCE_DIR})
set (librarysourcedir ${librarydir}/src)
set (libraryincludedir ${librarydir}/include)

set(sourceslibrary  ${librarysourcedir}/linearregression.cpp
             ${librarysourcedir}/matrixcode.cpp 
             ${librarysourcedir}/robustregression.cpp 
             ${librarysourcedir}/statisticfunctions.cpp 
             ${librarysourcedir}/nonlinearregression.cpp)

ADD_LIBRARY(RobustRegressionLib SHARED ${sourceslibrary})

target_include_directories(RobustRegressionLib PUBLIC  ${libraryincludedir} ${CMAKE_CURRENT_BINARY_DIR})

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} /std:c++17 /EHsc /openmp:experimental")
endif()


GENERATE_EXPORT_HEADER(RobustRegressionLib           
    BASE_NAME RobustRegressionLib
    EXPORT_MACRO_NAME ROBUSTREGRESSION_API
    EXPORT_FILE_NAME robustregressionlib_exports.h
    STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)

if(${WITH_PYTHON})
    set (librarypythondir ${librarysourcedir}/pyRobustRegressionLib)
    find_package(Python3 COMPONENTS Interpreter Development)  
    find_package(pybind11 CONFIG REQUIRED )

    set_target_properties(Python3::Module PROPERTIES MAP_IMPORTED_CONFIG_DEBUG ";RELEASE")

    pybind11_add_module(pyRobustRegressionLib ${librarypythondir}/pyRobustRegressionLib.cpp)

    target_include_directories(pyRobustRegressionLib PUBLIC ${libraryincludedir} ${CMAKE_CURRENT_BINARY_DIR})


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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} /std:c++17 /EHsc /openmp:experimental")
    endif()

   target_link_libraries(pyRobustRegressionLib PUBLIC RobustRegressionLib)

   if(DEFINED SKBUILD)
        install(TARGETS RobustRegressionLib DESTINATION lib)
        install(TARGETS pyRobustRegressionLib DESTINATION lib)
    endif()

endif()



