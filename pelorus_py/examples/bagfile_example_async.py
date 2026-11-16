#!/usr/bin/env python3
"""\
Example using Pelorus Python bindings with MCAP bagfile data (async callback).

This example reads IMU and lidar data from an MCAP bagfile and processes it
through Pelorus to produce odometry estimates.

Unlike bagfile_example.py (blocking recv), this script uses the async callback
API. It waits for the odometry corresponding to a specific input scan.

Requirements:
    pip install pelorus mcap mcap-ros2-support

Usage:
    python bagfile_example_async.py bagfile.mcap
"""

import sys
import threading
from pathlib import Path

import pelorus
from mcap_ros2.reader import read_ros2_messages


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bagfile_example_async.py <bagfile.mcap>")
        print()
        raise SystemExit(1)

    bagfile_path = Path(sys.argv[1])
    if not bagfile_path.exists():
        print(f"Error: Bagfile not found: {bagfile_path}")
        raise SystemExit(1)

    print(f"Pelorus Python Bindings v{pelorus.Handle.version()}")
    print(f"Reading bagfile: {bagfile_path}")
    print()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "velodyne.rl"

    lio = pelorus.Handle(
        str(config_path),
        device_specific_pointcloud2=True,
    )

    imu_topic = "/imu/data"
    lidar_topic = "/velodyne_points"

    imu_count = 0
    lidar_count = 0

    max_lidar_scans = 50

    completion = threading.Condition()
    odom_count = 0
    received_scan_ids = set()
    target_scan_id = None
    previous_scan_id = None

    def on_odom(odom: dict) -> None:
        nonlocal odom_count

        with completion:
            odom_count += 1
            current = odom_count
            received_scan_ids.add(odom["scan_id"])
            completion.notify_all()

        print(
            f"  Odom #{current}: pos=[{odom['position'][0]:.3f}, {odom['position'][1]:.3f}, {odom['position'][2]:.3f}] "
            f"vel=[{odom['linear_velocity'][0]:.3f}, {odom['linear_velocity'][1]:.3f}, {odom['linear_velocity'][2]:.3f}] "
            f"points={len(odom['points'])}"
        )

    # Set the callback to receive odometry results asynchronously.
    lio.set_callback(on_odom)

    print(f"Processing messages from topics: {imu_topic}, {lidar_topic}")
    print()

    with open(bagfile_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = msg.channel.topic

            if topic == imu_topic:
                lio.add_imu_ros(msg.ros_msg)
                imu_count += 1

            elif topic == lidar_topic:
                scan_id = lio.add_pointcloud2_ros(msg.ros_msg)
                target_scan_id = previous_scan_id
                previous_scan_id = scan_id
                lidar_count += 1
                
                if lidar_count >= max_lidar_scans:
                    break

    if target_scan_id is None:
        lio.clear_callback()
        raise RuntimeError("The bag must contain at least two lidar scans")

    # The final lidar message is a support scan for the target immediately
    # before it. Wait for that exact result; no wall-clock timeout is needed.
    with completion:
        while target_scan_id not in received_scan_ids:
            completion.wait()

    # Disable callback delivery before printing the summary.
    lio.clear_callback()

    with completion:
        final_odom_count = odom_count

    print()
    print("Summary:")
    print(f"  IMU messages processed: {imu_count}")
    print(f"  Lidar scans processed: {lidar_count}")
    print(f"  Odometry results received: {final_odom_count}")

if __name__ == "__main__":
    main()
