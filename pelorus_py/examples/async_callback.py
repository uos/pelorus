#!/usr/bin/env python3

import os
import pelorus
import numpy as np
import threading

DUMP_ALL = os.getenv("PELORUS_DUMP_ALL_POINTS") is not None


def main():
    print("Version:", pelorus.__version__)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." , "pelorus_c", "examples", "no-accum.rl"))
    lio = pelorus.Handle(config_path=config_path, device_specific_pointcloud2=False)

    done_event = threading.Event()
    matched_odom = {}
    target_scan_id = None

    def on_odom(odom):
        pts = odom.get('points', [])
        if odom.get('timestamp_sec', 0) == 0 and odom.get('timestamp_nanosec', 0) == 0:
            return
        if not pts or target_scan_id is None or odom.get('scan_id', 0) != target_scan_id:
            return

        print(f"Callback invoked! Odometry #1 received (scan_id={odom.get('scan_id',0)})")
        print(f"  Timestamp: {odom['timestamp_sec']}.{odom['timestamp_nanosec']:09d}")
        pos = odom['position']
        print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        lv = odom['linear_velocity']
        print(f"  Linear velocity: [{lv[0]:.3f}, {lv[1]:.3f}, {lv[2]:.3f}]")
        print(f"  Point cloud: {len(pts)} points")

        matched_odom['odom'] = odom
        done_event.set()
        if DUMP_ALL:
            for i, p in enumerate(pts):
                print("    Point %d: [%.6g, %.6g, %.6g] I=%.6g" % (i, p[0], p[1], p[2], p[3]))
        else:
            for i in range(min(3, len(pts))):
                p = pts[i]
                print("    Point %d: [%.2f, %.2f, %.2f] I=%.1f" % (i, p[0], p[1], p[2], p[3]))
        print()

    lio.set_callback(on_odom)

    base_sec = 1000

    # send 200 IMUs
    for i in range(200):
        lio.add_imu(
            timestamp_sec=base_sec,
            timestamp_nanosec=i * 5000000,
            orientation=[1.0, 0.0, 0.0, 0.0],
            orientation_covariance=[0.01] + [0.0] * 8,
            angular_velocity=[0.001 * (i % 3 - 1), 0.001 * ((i + 1) % 3 - 1), 0.002 * ((i + 2) % 3 - 1)],
            angular_velocity_covariance=[0.001] + [0.0] * 8,
            linear_acceleration=[0.01 * (i % 5 - 2), 0.01 * ((i + 1) % 5 - 2), 9.81 + 0.01 * ((i + 2) % 5 - 2)],
            linear_acceleration_covariance=[0.01] + [0.0] * 8,
        )

    # Scan 1: Nx4 [x,y,z,t]
    NUM_POINTS = 64
    idx = np.arange(NUM_POINTS, dtype=np.float32)
    angles = idx / NUM_POINTS * 6.28318
    rng = 5.0 + 0.5 * (idx % 8)
    scan1 = np.empty((NUM_POINTS, 4), dtype=np.float32)
    scan1[:, 0] = rng * np.cos(angles)
    scan1[:, 1] = rng * np.sin(angles)
    scan1[:, 2] = 0.2 * np.sin(angles * 3)
    scan1[:, 3] = (idx / NUM_POINTS * 100000000.0).astype(np.float32)

    scan1_id = lio.add_pointcloud(scan1, timestamp_sec=base_sec, timestamp_nanosec=600000000)
    if scan1_id == 0:
        print("Failed to add first point cloud")
    else:
        print(f"Added first scan id={scan1_id} with {NUM_POINTS} points")

    # Send more IMUs
    for i in range(200, 300):
        lio.add_imu(
            timestamp_sec=base_sec,
            timestamp_nanosec=i * 5000000,
            orientation=[1.0, 0.0, 0.0, 0.0],
            orientation_covariance=[0.01] + [0.0] * 8,
            angular_velocity=[0.001 * (i % 3 - 1), 0.001 * ((i + 1) % 3 - 1), 0.002 * ((i + 2) % 3 - 1)],
            angular_velocity_covariance=[0.001] + [0.0] * 8,
            linear_acceleration=[0.01 * (i % 5 - 2), 0.01 * ((i + 1) % 5 - 2), 9.81 + 0.01 * ((i + 2) % 5 - 2)],
            linear_acceleration_covariance=[0.01] + [0.0] * 8,
        )

    # Scan 2: Nx5 [x,y,z,intensity,t]
    idx2 = np.arange(NUM_POINTS, dtype=np.float32)
    angles2 = idx2 / NUM_POINTS * 6.28318 + 0.1
    rng2 = 5.0 + 0.5 * (idx2 % 8) + 0.05
    scan2 = np.empty((NUM_POINTS, 5), dtype=np.float32)
    scan2[:, 0] = rng2 * np.cos(angles2)
    scan2[:, 1] = rng2 * np.sin(angles2)
    scan2[:, 2] = 0.2 * np.sin(angles2 * 3)
    scan2[:, 3] = 0.0
    scan2[:, 4] = (idx2 / NUM_POINTS * 100000000.0).astype(np.float32)

    scan2_id = lio.add_pointcloud(scan2, timestamp_sec=base_sec + 1, timestamp_nanosec=200000000)
    if scan2_id == 0:
        print("Failed to add second point cloud")
        return 1
    target_scan_id = scan2_id

    if not done_event.wait(1.0):
        print("Timed out waiting for callback for scan_id", target_scan_id)
        lio.clear_callback()
        return 1

    lio.clear_callback()
    print("Matched odom:")
    if matched_odom:
        print(matched_odom['odom'])
    return 0

if __name__ == '__main__':
    main()
