cmake_minimum_required(VERSION 3.20)
project(arxiv_private_storage LANGUAGES CXX)

if(NOT EXISTS "${NODE_INCLUDE_DIR}/node_api.h")
  message(FATAL_ERROR "NODE_INCLUDE_DIR must point to the Node-API headers")
endif()

add_library(private_storage SHARED private-storage.cc)
target_include_directories(private_storage PRIVATE "${NODE_INCLUDE_DIR}")
target_compile_definitions(private_storage PRIVATE NAPI_VERSION=8)
target_compile_features(private_storage PRIVATE cxx_std_17)
set_target_properties(private_storage PROPERTIES
  PREFIX "" SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

if(MSVC)
  set_property(TARGET private_storage PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(NOT EXISTS "${NODE_LIBRARY}")
    message(FATAL_ERROR "NODE_LIBRARY must point to the matching Windows node.lib")
  endif()
  target_sources(private_storage PRIVATE win-delay-load.cc)
  target_compile_options(private_storage PRIVATE /W4 /EHsc)
  target_compile_definitions(private_storage PRIVATE _WIN32_WINNT=0x0A00)
  target_link_libraries(private_storage PRIVATE "${NODE_LIBRARY}" delayimp advapi32)
  target_link_options(private_storage PRIVATE /DELAYLOAD:node.exe)
else()
  target_compile_options(private_storage PRIVATE -Wall -Wextra -Werror -fvisibility=hidden)
  if(APPLE)
    target_link_options(private_storage PRIVATE "-undefined" "dynamic_lookup")
    set_target_properties(private_storage PROPERTIES OSX_DEPLOYMENT_TARGET "11.0")
  else()
    target_link_options(private_storage PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endif()
