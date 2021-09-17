cmake_minimum_required(VERSION 3.10)

project(DM_Cluster)

add_subdirectory(third_party/pybind11)

pybind11_add_module(DM_Cluster DM_Cluster.cpp)
