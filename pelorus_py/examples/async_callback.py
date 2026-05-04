#!/usr/bin/env python3

import os
import struct
import sys
import threading

import numpy as np
import pelorus

TAG_IMU = 1
TAG_CLOUD = 2
DUMP_ALL = os.getenv("PELORUS_DUMP_ALL_POINTS") is not None


def split_time(t):
    total_ns = int(round(t * 1_000_000_000))
    return total_ns // 1_000_000_000, total_ns % 1_000_000_000


def read_records(path):
    with open(path, "rb") as f:
        while True:
            tag = f.read(1)
            if not tag:
                return
            tag = tag[0]
            if tag == TAG_IMU:
                payload = f.read(8 + 6 * 4)
                if len(payload) != 8 + 6 * 4:
                    raise ValueError("truncated IMU record")
                t, gx, gy, gz, ax, ay, az = struct.unpack("<dffffff", payload)
                yield ("imu", t, (gx, gy, gz), (ax, ay, az))
            elif tag == TAG_CLOUD:
                header = f.read(8 + 8 + 4)
                if len(header) != 20:
                    raise ValueError("truncated cloud header")
                start, end, n = struct.unpack("<ddi", header)
                payload = f.read(n * 5 * 4)
                if len(payload) != n * 5 * 4:
                    raise ValueError("truncated cloud points")
                cloud = np.frombuffer(payload, dtype="<f4").reshape((n, 5)).copy()
                cloud[:, 4] *= 1_000_000_000.0
                yield ("cloud", start, end, cloud)
            else:
                raise ValueError(f"unknown record tag {tag}")


def main():
    print("Version:", pelorus.__version__)
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pelorus_c", "examples"))
    config_path = os.path.join(examples_dir, "no-accum.rl")
    dataset_path = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(examples_dir, "synthetic_dataset.bin")
    lio = pelorus.Handle(config_path=config_path, device_specific_pointcloud2=False)

    done_event = threading.Event()
    matched_odom = {}
    target_scan_id = None

    def on_odom(odom):
        pts = odom.get("points", None)
        if pts is None:
            return
        pts = np.asarray(pts)
        if odom.get("timestamp_sec", 0) == 0 and odom.get("timestamp_nanosec", 0) == 0:
            return
        if pts.size == 0 or target_scan_id is None or odom.get("scan_id", 0) != target_scan_id:
            return

        pos = np.asarray(odom["position"], dtype=np.float64)
        lv = np.asarray(odom["linear_velocity"], dtype=np.float64)
        orient = np.asarray(odom["orientation"], dtype=np.float64)
        av = np.asarray(odom["angular_velocity"], dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)

        print(f"Callback invoked! Odometry #1 received (scan_id={odom.get('scan_id', 0)})")
        print(f"  Timestamp: {odom['timestamp_sec']}.{odom['timestamp_nanosec']:09d}")
        print(f"  Position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        print(f"  Linear velocity: [{lv[0]:.6f}, {lv[1]:.6f}, {lv[2]:.6f}]")
        print(f"  Point cloud: {len(pts)} points")

        matched_odom["odom"] = {
            "timestamp_sec": odom["timestamp_sec"],
            "timestamp_nanosec": odom["timestamp_nanosec"],
            "scan_id": odom.get("scan_id", None),
            "position": pos,
            "orientation": orient,
            "linear_velocity": lv,
            "angular_velocity": av,
            "points": pts,
        }
        done_event.set()
        limit = len(pts) if DUMP_ALL else min(3, len(pts))
        for i in range(limit):
            p = pts[i]
            if DUMP_ALL:
                print("    Point %d: [%.6g, %.6g, %.6g] I=%.6g" % (i, p[0], p[1], p[2], p[3]))
            else:
                print("    Point %d: [%.2f, %.2f, %.2f] I=%.1f" % (i, p[0], p[1], p[2], p[3]))
        print()

    lio.set_callback(on_odom)

    for record in read_records(dataset_path):
        if record[0] == "imu":
            _, t, gyro, accel = record
            sec, nsec = split_time(t)
            lio.add_imu(
                timestamp_sec=sec,
                timestamp_nanosec=nsec,
                orientation=[1.0, 0.0, 0.0, 0.0],
                orientation_covariance=[0.01] + [0.0] * 8,
                angular_velocity=list(gyro),
                angular_velocity_covariance=[0.001] + [0.0] * 8,
                linear_acceleration=list(accel),
                linear_acceleration_covariance=[0.01] + [0.0] * 8,
            )
        else:
            _, start, _end, cloud = record
            sec, nsec = split_time(start)
            scan_id = lio.add_pointcloud(cloud, timestamp_sec=sec, timestamp_nanosec=nsec)
            if scan_id == 0:
                print("Failed to add point cloud")
                return 1
            target_scan_id = scan_id
            print(f"Added scan id={scan_id} with {len(cloud)} points")

    if not done_event.wait(1.0):
        print("Timed out waiting for callback for scan_id", target_scan_id)
        lio.clear_callback()
        return 1

    lio.clear_callback()
    print("Matched odom:")
    if matched_odom:
        print(matched_odom["odom"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
