cmake_minimum_required(VERSION 3.14)

project(GaussQuad VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

##
## MAIN_PROJECT CHECK
## determine if GaussQuad is built as a subproject (using add_subdirectory) or as the main project
##
set(MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MAIN_PROJECT ON)
endif()

##
## OPTIONS
##
option(GAUSSQUAD_INCLUDE_TESTS  "Build the GoogleTest unit tests" ${MAIN_PROJECT})
option(GAUSSQUAD_BUILD_EXAMPLES "Build the example programs"       ${MAIN_PROJECT})
option(GAUSSQUAD_INSTALL        "Generate install and package-config rules" ${MAIN_PROJECT})

##
## TARGET
## header-only INTERFACE library.
##
add_library(GaussQuad INTERFACE)
add_library(GaussQuad::GaussQuad ALIAS GaussQuad)
# Backwards-compatibility alias for in-tree Thermal-FIST consumers that still
# link against the old target name. New code should use GaussQuad::GaussQuad.
add_library(TFNumericalIntegration ALIAS GaussQuad)
target_compile_features(GaussQuad INTERFACE cxx_std_11)
target_include_directories(GaussQuad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

##
## EXAMPLES
##
if(GAUSSQUAD_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

##
## TESTS (GoogleTest, fetched on demand)
##
if(GAUSSQUAD_INCLUDE_TESTS)

  include(FetchContent)
  FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
      # Use the archive's timestamps on extraction (silences CMP0135 dev warning on CMake >= 3.24).
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Do not let GoogleTest add its headers/libs/config to our install tree.
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_subdirectory(test)

endif()

##
## INSTALL / PACKAGE CONFIG
## After installation, downstream projects can do:
##   find_package(GaussQuad REQUIRED)
##   target_link_libraries(app PRIVATE GaussQuad::GaussQuad)
##
if(GAUSSQUAD_INSTALL)
  include(CMakePackageConfigHelpers)

  set(GAUSSQUAD_CMAKE_CONFIG_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GaussQuad)

  # Install the interface target into an export set.
  install(TARGETS GaussQuad EXPORT GaussQuadTargets)

  # Install the public headers (the include/gaussquad/ tree).
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.h")

  # Export the targets under the GaussQuad:: namespace.
  install(EXPORT GaussQuadTargets
          FILE GaussQuadTargets.cmake
          NAMESPACE GaussQuad::
          DESTINATION ${GAUSSQUAD_CMAKE_CONFIG_DESTINATION})

  # Generate and install the package config + version files.
  configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GaussQuadConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfig.cmake
      INSTALL_DESTINATION ${GAUSSQUAD_CMAKE_CONFIG_DESTINATION})

  write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMajorVersion)

  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfigVersion.cmake
      DESTINATION ${GAUSSQUAD_CMAKE_CONFIG_DESTINATION})
endif()
