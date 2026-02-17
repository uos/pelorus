#!/bin/bash
# assuming sourced ros environment and starting from the package root directory
# creates the deb in the current dir

apt-get update -y

# needed for creating the deb
apt-get install -y debhelper sed grep

# deps from the package.xml
apt-get install -y cargo libclang-dev python3-catkin-pkg-modules

cargo_ver=$(grep -Po '^\s*version\s*=\s*"\K[^"]+' Cargo.toml)
xml_ver=$(grep -Po '<\s*version\s*>\K[^<]+' package.xml)

if [[ "$cargo_ver" != "$xml_ver" ]]; then
    echo "Versions differ!" >&2
    echo "   Cargo.toml  : $cargo_ver" >&2
    echo "   package.xml : $xml_ver" >&2
    exit 1
fi

shopt -s dotglob nullglob
mkdir pelorus-src
mv -- * pelorus-src/
shopt -u dotglob nullglob

if [ "${PELORUS_CUDA:-}" = 1 ]; then
    tar -cJf "ros-${ROS_DISTRO}-pelorus-cuda${CUDA_PART}_${xml_ver}.orig.tar.xz" pelorus-src
else
    tar -cJf "ros-${ROS_DISTRO}-pelorus_${xml_ver}.orig.tar.xz" pelorus-src
fi

cd pelorus-src

bloom-generate rosdebian

if [ "${PELORUS_CUDA:-}" = 1 ]; then
    # ignore fancy dependency things that break CUDA, just yolo it, the user is responsible for having the correct libcuda.so
    perl -pi -e 's/^(\s*dh_shlibdeps\b[^\n]*)/$1 || true/' debian/rules
fi

dpkg-buildpackage -us -uc

