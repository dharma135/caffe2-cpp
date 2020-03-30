file(GLOB PROTOBUF_DEFINITION_FILES "*.proto")
MESSAGE( STATUS "PROTOS:" ${PROTOBUF_DEFINITION_FILES})
set(caffe2proto "/project/SAS_xfer/data_hub/reddy/Work2019/MLNN/pytorch-latest")
foreach(file ${PROTOBUF_DEFINITION_FILES})
    MESSAGE(STATUS "Compiling:" ${file})
    #set(PROTOBUF_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/util")
    #set(PROTOBUF_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/util")
    get_filename_component(PROTOBUF_INPUT_DIRECTORY ${file} DIRECTORY)
    set(PROTOBUF_OUTPUT_DIRECTORY "${PROTOBUF_INPUT_DIRECTORY}")
    get_filename_component(ABS_FIL ${file} ABSOLUTE)
    get_filename_component(FIL_WE ${file} NAME_WE)
    set(PROTOBUF_ARGUMENTS " --proto_path=${PROTOBUF_INPUT_DIRECTORY} --proto_path=${caffe2proto} --cpp_out=${PROTOBUF_OUTPUT_DIRECTORY} ${FIL_WE}.proto")
    MESSAGE(STATUS "cmd:" ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTOBUF_ARGUMENTS})
    #execute_process(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTOBUF_ARGUMENTS}
    #        WORKING_DIRECTORY ${PROTOBUF_INPUT_DIRECTORY}
    #        RESULT_VARIABLE PROTOBUF_RESULT
    #        OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE
    #        ERROR_VARIABLE PROTOBUF_ERROR_VARIABLE)
    exec_program(${PROTOBUF_PROTOC_EXECUTABLE} ${PROTOBUF_INPUT_DIRECTORY}
    ARGS ${PROTOBUF_ARGUMENTS}
    OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE
    RESULT_VARIABLE PROTOBUF_RESULT
    )
    exec_program(sed ${PROTOBUF_INPUT_DIRECTORY} ARGS "-i 's/caffe2.pb.h/caffe2_pb.h/g' ${FIL_WE}.pb.h")
    #   MESSAGE( STATUS "CMD_ERROR:" ${PROTOBUF_ERROR_VARIABLE})
            MESSAGE( STATUS "CMD_OUTPUT:" ${PROTOBUF_OUTPUT_VARIABLE})
            MESSAGE( STATUS "CMD_RESULT:" ${PROTOBUF_RESULT})
endforeach()

#file(GLOB PROTOBUF_MODELS_INCLUDES "src/util/*.pb.cc" "src/util/*.pb.h")

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS external.proto model.proto)
#include_directories("/project/SAS_xfer/data_hub/reddy/Work2019/MLNN/caffe2-cpp/build/src/util")
#message("generated protofiles ${PROTO_SRCS} ${PROTO_HDRS} ${CMAKE_CURRENT_BINARY_DIR}")