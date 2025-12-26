#/bin/bash

set -e

sudo apt-get install lsb-release --assume-yes

if [ "$(lsb_release -rs)" != "24.04" ]; then
    echo "This script is for Ubuntu 24.04"
    exit 1
fi

sudo apt-get update --assume-yes && sudo apt-get install locales --assume-yes
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
#echo "export LANG=en_US.UTF-8" >> ~/.bashrc
#echo "export LANG=en_US.UTF-8" >> ~/.zshrc

export LANG=en_US.UTF-8

echo "######"
echo "Check if your .bashrc/.zshrc has 'export LANG=en_US.UTF-8' and add it if it does not exist."
read -p "Press Enter when done..." tmp

# Zsh and tools
sudo apt-get install software-properties-common --assume-yes

sudo add-apt-repository ppa:neovim-ppa/unstable -y
sudo apt-get update --assume-yes
sudo apt-get install neovim --assume-yes

sudo apt-get install zsh zsh-autosuggestions zsh-syntax-highlighting tmux nano curl python3-pip --assume-yes
chsh -s /usr/bin/zsh

# Languages
## Go
sudo add-apt-repository ppa:longsleep/golang-backports -y
sudo apt-get update --assume-yes
sudo apt-get install golang-go --assume-yes

## Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup default stable

# ROS2 Jazzy
sudo add-apt-repository universe -y
sudo apt-get update --assume-yes
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update --assume-yes && sudo apt-get install ros-dev-tools ros-jazzy-desktop --assume-yes

sudo apt-get install libclang-dev --assume-yes
cargo install cargo-binstall

echo "Installed ROS2 Jazzy Environment"

# ---

echo "Installing GLIM"

sudo apt-get install -y curl gpg

curl -s https://koide3.github.io/ppa/setup_ppa.sh | sudo bash
sudo apt-get update -y
sudo apt-get install -y libiridescence-dev libboost-all-dev libglfw3-dev libmetis-dev libgtsam-points-dev ros-jazzy-glim-ros

# ---

# Pelorus
# assume pelorus is already there

mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

## GLIM Frontend
cd ~/ros2_ws/src/pelorus/glim_integration/pelorus_glim_frontend
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install

## minimal config
cd ~/ros2_ws/src/pelorus/glim_integration/
cat << 'EOF' > pelorus.rl
registration {

  map {
    resolution = 0.5m # Voxel side length. NOTE: This will heavily affect your registration accuracy.
    capacity = 100000 # How many Voxels of the previous defined resolution to save at any time. With Cuda enabled, the capacity will be round up to the next power of 2.
    
    distribution_update_sensitivity = 10.0 # Magic weighting value for handling the sensivity to new scans. The higher this value, the sensitive is it to brand new scans.
  }
}

lidar {

  vendor = Ouster # Velodyne | Ouster | Robosense | Skip (Use any 3D sensor when just Lidar Odometry is selected)

  covariance = 0.001 # Weight for ICP correction in the IEKF
}


imu {
  initial_setup = 0.1s

  from_lidar_transform {
    # translation = [-6.253mm, 11.775mm, 28.535mm] # Translation Vector Lidar_to_IMU
    translation = [0.0m, 0.0m, 0m] # Translation Vector Lidar_to_IMU

    # Rotation Matrix Lidar_to_IMU
    rotation = [ [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, 1.0] ]
  }

  ahrs_filtering.method = FilterAccelerationAndCreateOrientation # CreateOrientationOnly | FilterAccelerationAndCreateOrientation | FilterAccelerationOnly | Disable (recommended when using on-device filtering in the driver)

  covariance_gyroscope = 0.1
  covariance_gyroscope_bias = 0.0001

  covariance_acceleration = 0.1
  covariance_acceleration_bias = 0.0001


}

preprocessing.downsample.resolution = 25cm
EOF

sudo tee /opt/ros/jazzy/share/glim/config/config_pelorus_frontend.json << EOF
{
  "pelorus_frontend": {
    "pelorus_rl_path": "$HOME/ros2_ws/src/pelorus/glim_integration/pelorus.rl",
    "stamp_match_tolerance_sec": 0.02,

    "between_sigma_trans": 0.2,
    "between_sigma_rot_rad": 0.15,

    "max_frame_buffer": 400
  }
}
EOF


## GLIM Odom
cd ~/ros2_ws/src/pelorus/glim_integration/pelorus_glim_odometry
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install

