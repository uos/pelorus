#!/usr/bin/env python3
import os
import time
import pelorus
import numpy as np
import threading
import sys

DUMP_ALL = os.getenv("PELORUS_DUMP_ALL_POINTS") is not None

count = 0
target_scan_id = 0
_done_event = None
_last_scan_id = 0
_last_scan_lock = threading.Lock()
_expected_ts = None  # (sec, nanosec) tuple
_expected_num_points = None
_expected_first_point = None  # (x,y,z) tuple
_matched_odom = None  # store matched odom dict for main to print/inspect


def on_odom(odom):
    global count, _done_event, _last_scan_id
    # DEBUG: indicate that callback was invoked (helps detect races)
    try:
        print(f"DEBUG: on_odom called (scan_id={odom.get('scan_id',0)}, pts={len(odom.get('points',[]))}, ts={odom.get('timestamp_sec',0)}.{odom.get('timestamp_nanosec',0):09d})")
    except Exception:
        print("DEBUG: on_odom called (malformed odom)")
    # Ignore zero/placeholder odometry (seen during initialization)
    if odom.get('timestamp_sec', 0) == 0 and odom.get('timestamp_nanosec', 0) == 0:
        return
    # Ignore empty point clouds (we only care about the registered scan result)
    pts = odom.get('points', [])
    if not pts:
        return

    scan_id = odom.get('scan_id', 0)
    # Record latest scan id (thread-safe)
    with _last_scan_lock:
        _last_scan_id = scan_id

    # If we have an expected timestamp and number of points, check them (avoids relying on scan_id)
    if _expected_ts is not None and _expected_num_points is not None:
        exp_sec, exp_nsec = _expected_ts
        # exact match
        if (odom.get('timestamp_sec') == exp_sec and
            odom.get('timestamp_nanosec') == exp_nsec and
            len(pts) == _expected_num_points):
            matched = True
        else:
            # allow timestamp tolerance (e.g., small scheduling or undistort differences)
            try:
                odom_ts = float(odom.get('timestamp_sec', 0)) + float(odom.get('timestamp_nanosec', 0)) / 1e9
                exp_ts = float(exp_sec) + float(exp_nsec) / 1e9
                if abs(odom_ts - exp_ts) < 0.25 and len(pts) == _expected_num_points:
                    matched = True
                else:
                    matched = False
            except Exception:
                matched = False

        # Additionally, if we have an expected first point, try a quick geometric match
        if not matched and _expected_first_point is not None and len(pts) >= 1:
            fx, fy, fz = _expected_first_point
            p = pts[0]
            # tolerance chosen small; geometry is deterministic here
            if (abs(p[0] - fx) < 1e-3 and abs(p[1] - fy) < 1e-3 and abs(p[2] - fz) < 1e-3):
                matched = True

        # Also accept direct scan_id equality as a final fallback
        if not matched and target_scan_id != 0 and scan_id == target_scan_id:
            matched = True
    else:
        # Fallback: if a target_scan_id is set, match by it
        matched = (target_scan_id != 0 and scan_id == target_scan_id)

    if not matched:
        return

    count += 1
    print(f"Callback invoked! Odometry #{count} received (scan_id={scan_id})")
    print(f"  Timestamp: {odom['timestamp_sec']}.{odom['timestamp_nanosec']:09d}")
    pos = odom['position']
    print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    lv = odom['linear_velocity']
    print(f"  Linear velocity: [{lv[0]:.3f}, {lv[1]:.3f}, {lv[2]:.3f}]")
    print(f"  Point cloud: {len(pts)} points")

    # Save matched odom for main thread to inspect if needed
    global _matched_odom
    _matched_odom = odom

    if _done_event is not None:
        _done_event.set()
    if DUMP_ALL:
        for i, p in enumerate(pts):
            # p is [x, y, z, intensity, timestamp]
            print("    Point %d: [%.6g, %.6g, %.6g] I=%.6g t=%.6g" % (i, p[0], p[1], p[2], p[3], p[4]))
    else:
        for i in range(min(3, len(pts))):
            p = pts[i]
            print("    Point %d: [%.2f, %.2f, %.2f] I=%.1f t=%.6f" % (i, p[0], p[1], p[2], p[3], p[4]))
    print()


def main():
    print("Version:", pelorus.__version__)
    lio = pelorus.Handle()

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
    else:
        print(f"Added second scan id={scan2_id} with {NUM_POINTS} points")
        # inform callback about the scan id we care about
        global target_scan_id, _done_event, _last_scan_id, _last_scan_lock, _expected_ts, _expected_num_points, _expected_first_point
        target_scan_id = scan2_id
        _done_event = threading.Event()
        _expected_ts = (base_sec + 1, 200000000)
        _expected_num_points = NUM_POINTS
        # record first point to allow a quick geometric match (robust to small timestamp differences)
        _expected_first_point = (float(scan2[0,0]), float(scan2[0,1]), float(scan2[0,2]))

        # If the callback already received this scan (race), honor it immediately
        with _last_scan_lock:
            if _last_scan_id == scan2_id:
                _done_event.set()

    # Wait on the event up to 10s (purely callback-driven, no recv)
    signaled = False
    if _done_event is not None:
        signaled = _done_event.wait(10.0)

    if not signaled:
        print("Timed out waiting for callback for scan_id", scan2_id)
        # If we received something non-matching earlier, still print it for debugging
        if _matched_odom is not None:
            print("Partial/mismatch odom seen:")
            try:
                print(_matched_odom)
            except Exception:
                pass
        # Exit non-zero to indicate failure
        sys.exit(1)
    else:
        # Normal path: callback handled it; clear callback now
        lio.clear_callback()
        # Print the matched odom summary then exit
        print("Matched odom:")
        try:
            print(_matched_odom)
        except Exception:
            pass
        sys.exit(0)

    print("Done")


if __name__ == '__main__':
    main()
