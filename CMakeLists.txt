cmake_minimum_required(VERSION 3.16)
project(llm_rag_examples CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CURL REQUIRED)

foreach(ex basic_rag chunker file_rag)
    add_executable(${ex} examples/${ex}.cpp)
    target_include_directories(${ex} PRIVATE include)
    target_link_libraries(${ex} CURL::libcurl)
    target_compile_options(${ex} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    )
endforeach()
