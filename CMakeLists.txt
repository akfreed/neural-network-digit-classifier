cmake_minimum_required (VERSION 3.5.2)

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Solution
project(NeuralNet CXX)

# Warning flags
if(MSVC)
    # Force to always compile with W4
    if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
    if (NOT CMAKE_CXX_FLAGS MATCHES "/WX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
    endif()
elseif (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()

# Enable AVX (Vectorized Instructions)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVX_SUPPORTED)
if (COMPILER_OPT_ARCH_AVX_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif (COMPILER_OPT_ARCH_AVX_SUPPORTED)


include_directories(
    eigen
)


# -------------------------------------------------------------------
# Projects

# NeuralNet
add_executable(NeuralNet
    src/FileIO.cpp
    src/FileIO.h
    src/main.cpp
    src/NeuralNet.cpp
    src/NeuralNet.h
    src/Trainer.h
    src/UnitTest.cpp
    src/UnitTest.h
    src/Utility.h
)
# set Visual Studio working directory
set_target_properties(NeuralNet PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
# add Natvis (VS) to source files
if (MSVC)
    target_sources(NeuralNet PRIVATE "eigen/debug/msvc/eigen.natvis")
endif (MSVC)


# Set NeuralNet as the start-up project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NeuralNet)
