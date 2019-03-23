cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
set(COMPILE_FLAGSO2 "-Xcicc -O0 -Xcudafe \"--diag_suppress=divide_by_zero\"")
include(CTest)
enable_testing()

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# CORE
add_library(libcu-sentinel.${arch} STATIC
  libcu-sentinel/_sentinel_int.cu
  libcu-sentinel/sentinel.cpp
  )
  set_source_files_properties(libcu-sentinel/_sentinel_int.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu-sentinel.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu-sentinel.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu-sentinel.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()


# TESTS
if (BUILD_TESTING)
  add_executable(libcu-sentinel_tests
	libcu-sentinel.tests/sentinelTest.cu
	libcu-sentinel.tests/program.cu
  )
  set_source_files_properties(libcu-sentinel.tests/libcu-sentinel.tests.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")

  set_target_properties(libcu-sentinel_tests PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
  if (M)
    set_target_properties(libcu-sentinel_tests PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64} CUDA_SEPARABLE_COMPILATION ON)
  endif()
  target_link_libraries(libcu-sentinel_tests PRIVATE libcu.fileutils.${arch})

  add_test(NAME sentinel_test1 COMMAND libcu-sentinel_tests 1)

  if (APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that the static cuda runtime can find it at runtime.
    set_property(TARGET libcu_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()