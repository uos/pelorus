#!/usr/bin/env python3
import re
import subprocess
import sys
from math import isclose

POINT_RE = re.compile(r"^\s*Point\s+(\d+):\s*\[([^]]+)\]\s*I=([0-9eE+\-\.]+)\s*t=([0-9eE+\-\.]+)")


def run_cmd(cmd, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, env=env)
    if p.returncode != 0:
        print(f"Command failed: {cmd}\nstdout:\n{p.stdout.decode()}\nstderr:\n{p.stderr.decode()}")
        sys.exit(2)
    return p.stdout.decode()


def parse_points(output):
    # find first occurrence of a header line (either callback or scan_id line) and then collect Point lines
    lines = output.splitlines()
    pts = []
    collecting = False
    for i, ln in enumerate(lines):
        if 'scan_id=' in ln or 'Point cloud:' in ln:
            # if we see a Point line soon after, start collecting
            # Note: we start collection when we see the first matching Point line
            pass
        m = POINT_RE.match(ln)
        if m:
            idx = int(m.group(1))
            coords = m.group(2).split(',')
            coords = [float(x.strip()) for x in coords]
            intensity = float(m.group(3))
            t = float(m.group(4))
            pts.append((coords[0], coords[1], coords[2], intensity, t))
            collecting = True
        else:
            if collecting:
                # stop on first non-point line after collecting started
                break
    return pts


def main():
    # Build examples first
    print("Building examples...")
    subprocess.check_call(["make", "-C", "."])

    env = dict(**subprocess.os.environ)
    env['PELORUS_DUMP_ALL_POINTS'] = '1'

    print("Running async_callback...")
    out_async = run_cmd("./async_callback", env=env)

    print("Running sync_blocking...")
    out_sync = run_cmd("./sync_blocking", env=env)

    # Build and run cpp_wrapper using CMake if possible; if it's not available, skip it.
    print("Attempting to build cpp_wrapper via CMake (may be skipped if PCL/Eigen missing)...")
    cpp_built = False
    try:
        # allow make to fail without aborting the whole script
        subprocess.run("make cpp_wrapper", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if subprocess.os.path.exists("build/cpp_wrapper"):
            print("Running cpp_wrapper (build/cpp_wrapper)...")
            out_cpp = run_cmd("./build/cpp_wrapper", env=env)
            cpp_built = True
        else:
            out_cpp = None
    except subprocess.CalledProcessError:
        print("cpp_wrapper build skipped or failed; continuing with async/sync comparison")
        out_cpp = None

    pts_async = parse_points(out_async)
    pts_sync = parse_points(out_sync)
    pts_cpp = parse_points(out_cpp) if out_cpp is not None else None

    if not pts_async:
        print("No points parsed from async output")
        print(out_async)
        sys.exit(2)
    if not pts_sync:
        print("No points parsed from sync output")
        print(out_sync)
        sys.exit(2)

    if len(pts_async) != len(pts_sync):
        print(f"Different point counts: async={len(pts_async)} sync={len(pts_sync)}")
        sys.exit(1)

    eps = 1e-4
    for i, (a, b) in enumerate(zip(pts_async, pts_sync)):
        for k in range(5):
            if not isclose(a[k], b[k], rel_tol=1e-6, abs_tol=eps):
                print(f"Point {i} differs between async and sync at element {k}: async={a[k]} sync={b[k]}")
                sys.exit(1)

    print("SUCCESS: async and sync point clouds match exactly within tolerance")

    if cpp_built:
        if not pts_cpp:
            print("cpp_wrapper produced no points; failing")
            print(out_cpp)
            sys.exit(2)
        if len(pts_cpp) != len(pts_async):
            print(f"Different point counts: async={len(pts_async)} cpp={len(pts_cpp)}")
            sys.exit(1)
        for i, (a, c) in enumerate(zip(pts_async, pts_cpp)):
            for k in range(5):
                if not isclose(a[k], c[k], rel_tol=1e-6, abs_tol=eps):
                    print(f"Point {i} differs between async and cpp at element {k}: async={a[k]} cpp={c[k]}")
                    sys.exit(1)
        print("SUCCESS: cpp_wrapper matches async/sync point clouds within tolerance")
    else:
        print("cpp_wrapper not built; skipped cpp comparison")

    if not pts_async:
        print("No points parsed from async output")
        print(out_async)
        sys.exit(2)
    if not pts_sync:
        print("No points parsed from sync output")
        print(out_sync)
        sys.exit(2)

    if len(pts_async) != len(pts_sync):
        print(f"Different point counts: async={len(pts_async)} sync={len(pts_sync)}")
        sys.exit(1)

    eps = 1e-4
    for i, (a, b) in enumerate(zip(pts_async, pts_sync)):
        for k in range(5):
            if not isclose(a[k], b[k], rel_tol=1e-6, abs_tol=eps):
                print(f"Point {i} differs at element {k}: async={a[k]} sync={b[k]}")
                sys.exit(1)

    print("SUCCESS: async and sync point clouds match exactly within tolerance")

if __name__ == '__main__':
    main()
