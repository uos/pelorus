#!/usr/bin/env python3
"""
Example using Pelorus Python bindings with MCAP bagfile data.

This example reads IMU and lidar data from an MCAP bagfile and processes it
through Pelorus to produce odometry estimates.

Requirements:
    pip install pelorus mcap mcap-ros2-support

Usage:
    python bagfile_example.py bagfile.mcap
"""

import sys
from pathlib import Path

import pelorus
from mcap_ros2.reader import read_ros2_messages

def main():
    if len(sys.argv) < 2:
        print("Usage: python bagfile_example.py <bagfile.mcap>")
        print()
        sys.exit(1)

    bagfile_path = Path(sys.argv[1])
    if not bagfile_path.exists():
        print(f"Error: Bagfile not found: {bagfile_path}")
        sys.exit(1)

    print(f"Pelorus Python Bindings v{pelorus.Handle.version()}")
    print(f"Reading bagfile: {bagfile_path}")
    print()

    # Since we have ROS2 pointcloud messages, we use the device-specific pointcloud2 option
    # to let Pelorus handle the timestamp internally. Also, we do not have ROS running with TF,
    # so the config file also indicates the static transforms.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir
    config_path = repo_root / "velodyne.rl"
    lio = pelorus.Handle(
        str(config_path),
        device_specific_pointcloud2=True,
    )

    imu_topic = "/imu/data"
    lidar_topic = "/velodyne_points"
    
    imu_count = 0
    lidar_count = 0
    odom_count = 0
    received_scan_ids = set()
    target_scan_id = None
    previous_scan_id = None

    def print_odom(odom):
        print(
            f"  Position: [{odom['position'][0]:.3f}, {odom['position'][1]:.3f}, {odom['position'][2]:.3f}]"
        )
        print(
            f"  Orientation (quat): [{odom['orientation'][0]:.3f}, {odom['orientation'][1]:.3f}, {odom['orientation'][2]:.3f}, {odom['orientation'][3]:.3f}]"
        )
        print(
            f"  Linear velocity: [{odom['linear_velocity'][0]:.3f}, {odom['linear_velocity'][1]:.3f}, {odom['linear_velocity'][2]:.3f}]"
        )
        print(f"  Point cloud: {len(odom['points'])} points")
        print()
    
    print(f"Processing messages from topics: {imu_topic}, {lidar_topic}")
    print()

    # The final scan is a support scan: its preceding IMU messages allow the
    # previous scan to finish before input stops.
    max_lidar_scans = 50

    with open(bagfile_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = msg.channel.topic
            
            if topic == imu_topic:
                imu = msg.ros_msg
                lio.add_imu_ros(imu)
                imu_count += 1
            
            elif topic == lidar_topic:
                cloud = msg.ros_msg
                scan_id = lio.add_pointcloud2_ros(cloud)
                target_scan_id = previous_scan_id
                previous_scan_id = scan_id
                lidar_count += 1
                
                # Drain odometry as we go so we don't let the internal queue grow unbounded.
                while True:
                    odom = lio.try_recv()
                    if odom is None:
                        break
                    odom_count += 1
                    received_scan_ids.add(odom["scan_id"])
                    print_odom(odom)

                if lidar_count >= max_lidar_scans:
                    break

    if target_scan_id is None:
        raise RuntimeError("The bag must contain at least two lidar scans")

    # The support scan above lets this specific target complete deterministically.
    while target_scan_id not in received_scan_ids:
        odom = lio.recv()
        odom_count += 1
        received_scan_ids.add(odom["scan_id"])
        print_odom(odom)

    print()
    print(f"Summary:")
    print(f"  IMU messages processed: {imu_count}")
    print(f"  Lidar scans processed: {lidar_count}")
    print(f"  Odometry results received: {odom_count}")

if __name__ == "__main__":
    main()
