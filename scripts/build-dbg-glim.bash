#!/bin/bash

set -x
set -o pipefail

cd glim_integration/pelorus_glim_odometry
mkdir -p build || true
cd build
cmake ..
make -j4
cd ../../..