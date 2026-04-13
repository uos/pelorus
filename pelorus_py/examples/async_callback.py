#!/usr/bin/env python3

import os
import pelorus
import numpy as np
import threading

DUMP_ALL = os.getenv("PELORUS_DUMP_ALL_POINTS") is not None


def main():
    print("Version:", pelorus.__version__)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." , "pelorus_c", "examples", "pico-equality.rl"))
    lio = pelorus.Handle(config_path=config_path, device_specific_pointcloud2=False)

    done_event = threading.Event()
    matched_odom = {}
    target_scan_id = None

    def on_odom(odom):
        pts = odom.get('points', None)
        if pts is None:
            return
        pts = np.asarray(pts)
        if odom.get('timestamp_sec', 0) == 0 and odom.get('timestamp_nanosec', 0) == 0:
            return
        if pts.size == 0 or target_scan_id is None or odom.get('scan_id', 0) != target_scan_id:
            return

        pos = np.asarray(odom['position'], dtype=np.float64)
        lv = np.asarray(odom['linear_velocity'], dtype=np.float64)
        orient = np.asarray(odom['orientation'], dtype=np.float64)
        av = np.asarray(odom['angular_velocity'], dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)

        print(f"Callback invoked! Odometry #1 received (scan_id={odom.get('scan_id',0)})")
        print(f"  Timestamp: {odom['timestamp_sec']}.{odom['timestamp_nanosec']:09d}")
        print(f"  Position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        print(f"  Linear velocity: [{lv[0]:.6f}, {lv[1]:.6f}, {lv[2]:.6f}]")
        print(f"  Point cloud: {len(pts)} points")

        matched_odom['odom'] = {
            'timestamp_sec': odom['timestamp_sec'],
            'timestamp_nanosec': odom['timestamp_nanosec'],
            'scan_id': odom.get('scan_id', None),
            'position': pos,
            'orientation': orient,
            'linear_velocity': lv,
            'angular_velocity': av,
            'points': pts,
        }
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
            angular_velocity=[0.004 + 0.001 * (i % 3 - 1), -0.002 + 0.001 * ((i + 1) % 3 - 1), 0.010 + 0.002 * ((i + 2) % 3 - 1)],
            angular_velocity_covariance=[0.001] + [0.0] * 8,
            linear_acceleration=[0.03 + 0.01 * (i % 5 - 2), -0.015 + 0.01 * ((i + 1) % 5 - 2), 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2)],
            linear_acceleration_covariance=[0.01] + [0.0] * 8,
        )

    # Scan 1: Nx4 [x,y,z,t]
    NUM_POINTS = 512
    idx = np.arange(NUM_POINTS, dtype=np.float32)
    half = NUM_POINTS // 2
    idx_a = idx[:half]
    idx_b = idx[half:] - half
    scan1 = np.empty((NUM_POINTS, 4), dtype=np.float32)
    u_a = np.mod(idx_a, 16)
    v_a = np.floor_divide(idx_a, 16)
    scan1[:half, 0] = 4.0 + 0.12 * u_a
    scan1[:half, 1] = -1.0 + 0.12 * v_a
    scan1[:half, 2] = 0.01 * np.mod(u_a + v_a, 2)

    u_b = np.mod(idx_b, 16)
    v_b = np.floor_divide(idx_b, 16)
    scan1[half:, 0] = 6.0 + 0.005 * np.mod(u_b + v_b, 2)
    scan1[half:, 1] = -1.0 + 0.12 * u_b
    scan1[half:, 2] = 0.10 + 0.12 * v_b
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
            angular_velocity=[0.004 + 0.001 * (i % 3 - 1), -0.002 + 0.001 * ((i + 1) % 3 - 1), 0.010 + 0.002 * ((i + 2) % 3 - 1)],
            angular_velocity_covariance=[0.001] + [0.0] * 8,
            linear_acceleration=[0.03 + 0.01 * (i % 5 - 2), -0.015 + 0.01 * ((i + 1) % 5 - 2), 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2)],
            linear_acceleration_covariance=[0.01] + [0.0] * 8,
        )

    # Scan 2: Nx5 [x,y,z,intensity,t]
    idx2 = np.arange(NUM_POINTS, dtype=np.float32)
    idx2_a = idx2[:half]
    idx2_b = idx2[half:] - half
    x = np.empty(NUM_POINTS, dtype=np.float32)
    y = np.empty(NUM_POINTS, dtype=np.float32)
    z = np.empty(NUM_POINTS, dtype=np.float32)
    u2_a = np.mod(idx2_a, 16)
    v2_a = np.floor_divide(idx2_a, 16)
    x[:half] = 4.0 + 0.12 * u2_a
    y[:half] = -1.0 + 0.12 * v2_a
    z[:half] = 0.01 * np.mod(u2_a + v2_a, 2)
    u2_b = np.mod(idx2_b, 16)
    v2_b = np.floor_divide(idx2_b, 16)
    x[half:] = 6.0 + 0.005 * np.mod(u2_b + v2_b, 2)
    y[half:] = -1.0 + 0.12 * u2_b
    z[half:] = 0.10 + 0.12 * v2_b
    c = 0.996802
    s = 0.079915
    scan2 = np.empty((NUM_POINTS, 5), dtype=np.float32)
    scan2[:, 0] = c * x - s * y + 0.18
    scan2[:, 1] = s * x + c * y - 0.12
    scan2[:, 2] = z + 0.03
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
