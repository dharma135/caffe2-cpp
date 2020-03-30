cmake_minimum_required(VERSION 2.5)
include("CMakePolicy.config")
project (caffe2-cpp)
option(USE_NATIVE_ARCH "Use -march=native" ON)
list( APPEND CMAKE_PREFIX_PATH "/home/dreddy/work/MLNN/pytorch-latest/torch")
list( APPEND CMAKE_PREFIX_PATH "/home/dreddy/localApps/pytorch-deps/glog")
list( APPEND CMAKE_PREFIX_PATH "/home/dreddy/localApps/pytorch-deps/gflags")
list( APPEND CMAKE_PREFIX_PATH "/home/dreddy/localApps/pytorch-deps/leveldb")
list( APPEND CMAKE_PREFIX_PATH "$ENV{GTEST_ROOT_DIR}")
#find_package(Torch CONFIG REQUIRED)
#set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
find_package(Protobuf REQUIRED)
message("Found protobuf dirs ${Protobuf_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIR}")
find_package(GTest  REQUIRED)
message("Found GTEST dirs ${GTEST_INCLUDE_DIRS}")
find_package(Caffe2 CONFIG REQUIRED)


message("Found include dirs ${CAFFE2_INCLUDE_DIRS}")
#message("Found libraries ${CAFFE2_MAIN_LIBS}")
#message("LD_LIB_PAT $ENV{LD_LIBRARY_PATH}")
#message("${${CAFFE2_MAIN_LIBS}}")
## include files from current project
##
## include protobuf
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories("/home/dreddy/work/MLNN/caffe2-cpp/build")
include_directories(${CAFFE2_INCLUDE_DIRS})
include_directories("include")
include_directories("src")
include_directories("thirdparty")
##include_directories("/home/dreddy/work/MLNN/pytorch/torch/include/caffe2/proto")
## inlcude gtest
include_directories("${GTEST_INCLUDE_DIRS}")
#protobuf_generate(OUT_VAR oVar PROTOS src/util/external.proto src/util/model.proto)

file(GLOB PROTOBUF_DEFINITION_FILES "*.proto")
MESSAGE( STATUS "PROTOS:" ${PROTOBUF_DEFINITION_FILES})
set(caffe2proto "/home/dreddy/work/MLNN/pytorch-latest/caffe2/proto")
foreach(file ${PROTOBUF_DEFINITION_FILES})
    MESSAGE(STATUS "Compiling:" ${file})
    #set(PROTOBUF_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/util")
    #set(PROTOBUF_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/util")
    get_filename_component(PROTOBUF_INPUT_DIRECTORY ${file} DIRECTORY)
    set(PROTOBUF_OUTPUT_DIRECTORY "${PROTOBUF_INPUT_DIRECTORY}")
    get_filename_component(ABS_FIL ${file} ABSOLUTE)
    get_filename_component(FIL_WE ${file} NAME_WE)
    set(PROTOBUF_ARGUMENTS " --proto_path=${PROTOBUF_INPUT_DIRECTORY} --proto-path=${caffe2proto} --cpp_out=${PROTOBUF_OUTPUT_DIRECTORY} ${FIL_WE}.proto")
    MESSAGE(STATUS "cmd:" ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTOBUF_ARGUMENTS})
    #execute_process(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTOBUF_ARGUMENTS}
    #        WORKING_DIRECTORY ${PROTOBUF_INPUT_DIRECTORY}
    #        RESULT_VARIABLE PROTOBUF_RESULT
    #        OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE
    #        ERROR_VARIABLE PROTOBUF_ERROR_VARIABLE)
    exec_program(${PROTOBUF_PROTOC_EXECUTABLE} ${PROJECT_SOURCE_DIR}
    ARGS ${PROTOBUF_ARGUMENTS}
    OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE
    RESULT_VARIABLE PROTOBUF_RESULT
    )
    #   MESSAGE( STATUS "CMD_ERROR:" ${PROTOBUF_ERROR_VARIABLE})
            MESSAGE( STATUS "CMD_OUTPUT:" ${PROTOBUF_OUTPUT_VARIABLE})
            MESSAGE( STATUS "CMD_RESULT:" ${PROTOBUF_RESULT})
endforeach()


add_subdirectory("src/util")
## include c10

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-dev")
set(LINK_LIBS)
macro(link_whole_archive lib)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND LINK_LIBS -Wl,-force_load,$<TARGET_FILE_NAME:${lib}> ${lib})
  elseif(MSVC)
    list(APPEND LINK_LIBS -WHOLEARCHIVE:$<TARGET_FILE_NAME:${lib}>)
  else()
    list(APPEND LINK_LIBS -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
  endif()
endmacro()
file(GLOB LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/util/*.cc" "${PROJECT_SOURCE_DIR}/src/operators/*.cc")

message("Sources... ${LIB_SOURCES}")
add_library(caffe2-cpp ${LIB_SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})
link_whole_archive(caffe2-cpp)

list(APPEND LINK_LIBS ${Caffe2_MAIN_LIBS})
if (DEFINED Caffe2_MODULES)
list(APPEND LINK_LIBS ${Caffe2_MODULES})
endif()
list(APPEND LINK_LIBS ${PROTOBUF_LIBRARY})
list(APPEND LINK_LIBS GTest::GTest)
list(APPEND LINK_LIBS GTest::Main)
#list(APPEND LINK_LIBS "/home/dreddy/localApps/googletest/1.8.1/lib64/libgtest.a")

#list(APPEND LINK_LIBS "/home/dreddy/localApps/googletest/1.8.1/lib64/libgtest_main.a")

function(add_deps targetName) 

#target_link_libraries(${targetName} ${Caffe2_MAIN_LIBS})

#if (DEFINED Caffe2_MODULES)
#  target_link_libraries(${targetName} ${Caffe2_MODULES})
#endif()
## add protobuf
#target_link_libraries(${targetName} ${PROTOBUF_LIBRARY})
## add gtest 
#target_link_libraries(${targetName} "/home/dreddy/localApps/googletest/1.8.1/lib64/libgtest.a")
#target_link_libraries(${targetName} "/home/dreddy/localApps/googletest/1.8.1/lib64/libgtest_main.a")
target_link_libraries(${targetName} ${LINK_LIBS} )
endfunction(add_deps)

function(add_binary binName srcs)
  add_executable(${binName} ${srcs} )
  add_deps(${binName})
  install (TARGETS ${binName} DESTINATION bin)
endfunction(add_binary)

## genrate proto files




add_binary(intro src/helloCaffe2.cc)
add_binary(testGEMM src/testGEMM.cc)
add_binary(testCNN src/testCNN.cc)
add_binary(testMNIST src/testMNIST.cc)
add_binary(testOP src/testOP.cc)
add_binary(testDAC src/test/testDAC.cc)
#PROTOBUF_GENERATE(LANGUAGE cpp TARGET testMNIST PROTOS  )
#add_executable(intro src/helloCaffe2.cc)
#add_executable(testGEMM src/testGEMM.cc)
#add_executable(testCNN src/testCNN.cc)
#add_deps(intro)
#add_deps(testGEMM)
#add_deps(testCNN)
#
#install(TARGETS intro DESTINATION bin)
#install(TARGETS testGEMM DESTINATION bin)
#install(TARGETS testCNN DESTINATION bin)

#message("Found include dirs ${CAFFE2_INCLUDE_DIRS}")
#message("Found libraries ${Caffe2_MAIN_LIBS}")

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
