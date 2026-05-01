#!/usr/bin/env bash
# Copy to local_env.sh (ignored by git) and edit for this machine:
#   cp gear_sonic_deploy/local_env.example.sh gear_sonic_deploy/local_env.sh
#   source gear_sonic_deploy/local_env.sh
#   cd gear_sonic_deploy && source scripts/setup_env.sh
#
# Keep this file free of secrets. It only documents machine-local paths used by
# the C++ deploy stack.

# Required before building gear_sonic_deploy. Point this at the extracted
# TensorRT directory that contains include/ and lib/.
export TensorRT_ROOT="${TensorRT_ROOT:-$HOME/TensorRT}"

# Optional: set if nvcc is not already on PATH. Pick the CUDA version that
# matches your TensorRT install and NVIDIA driver.
# export CUDAToolkit_ROOT="/usr/local/cuda-12.6"
# export CUDA_HOME="$CUDAToolkit_ROOT"
# export PATH="$CUDAToolkit_ROOT/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/lib64:$CUDAToolkit_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Optional: set if CMake cannot find the ONNX Runtime C/C++ package. This repo's
# custom finder expects an install root that contains include/ and lib/.
# export onnxruntime_ROOT="$HOME/.local/onnxruntime"
# export CMAKE_PREFIX_PATH="$onnxruntime_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# TensorRT runtime libraries.
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Real robot network reminder: the G1 control PC normally needs an interface on
# 192.168.123.x. Do not run `deploy.sh real` until sim mode is stable.
