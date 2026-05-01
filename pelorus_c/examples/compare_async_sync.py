#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from math import isclose

POINT_RE = re.compile(r"^\s*Point\s+(\d+):\s*\[([^]]+)\]\s*I=([0-9eE+\-\.]+)")


def log(msg):
    print(msg, flush=True)


def run_cmd(cmd, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, env=env)
    if p.returncode != 0:
        print(f"Command failed: {cmd}\nstdout:\n{p.stdout.decode()}\nstderr:\n{p.stderr.decode()}")
        sys.exit(2)
    return p.stdout.decode()


def run_cmd_allow_fail(cmd, env=None):
    """Run a command but don't sys.exit on non-zero return; return (stdout, stderr, returncode)"""
    p = subprocess.run(cmd, shell=True, capture_output=True, env=env)
    return (p.stdout.decode(), p.stderr.decode(), p.returncode)


def run_and_collect_until(cmd, env=None, sentinels=None):
    """Run command and stream stdout until one of the sentinel regexes appears.
    Returns (stdout, stderr, returncode). Does not use timeouts; determinism relies on the example to emit sentinel and exit."""
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    out_lines = []
    # Read stdout line-by-line until sentinel is matched
    matched = False
    should_stop_process = False
    terminated_by_us = False
    try:
        point_count = None
        point_lines = 0
        for ln in p.stdout:
            out_lines.append(ln)
            if sentinels and not matched:
                for s in sentinels:
                    if re.search(s, ln):
                        matched = True
                        break
            if matched:
                # look for point-cloud header to determine expected number of points
                m_pc = re.search(r"Point cloud: (\d+) points", ln)
                if m_pc:
                    try:
                        point_count = int(m_pc.group(1))
                    except Exception:
                        point_count = None
                # count point lines
                m_point = POINT_RE.match(ln)
                if m_point:
                    point_lines += 1
                    if point_count is not None and point_lines >= point_count:
                        # collected all points for the registered odom; we're done
                        should_stop_process = True
                        break
        if should_stop_process and p.poll() is None:
            terminated_by_us = True
            p.terminate()
            try:
                rest_out, rest_err = p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                rest_out, rest_err = p.communicate()
        else:
            # capture any remaining output after process exits
            try:
                rest_out, rest_err = p.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                p.kill()
                rest_out, rest_err = p.communicate()
        if rest_out:
            out_lines.append(rest_out)
        stderr = rest_err if rest_err else ""
    finally:
        # Ensure process always exits
        if p.poll() is None:
            p.kill()
            p.wait()
    rc = p.returncode
    if terminated_by_us and rc is not None and rc < 0:
        rc = 0
    return ("".join(out_lines), stderr, rc)


def parse_points(output):
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
            pts.append((coords[0], coords[1], coords[2], intensity))
            collecting = True
        else:
            if collecting:
                # stop on first non-point line after collecting started
                break
    return pts


def parse_registered_scan(output):
    """Extract the registered scan's scan_id and timestamp from output.

    Returns (scan_id (int or None), timestamp (float or None)).
    """
    if output is None:
        return (None, None)

    lines = output.splitlines()

    for ln in lines:
        if 'DEBUG: on_odom called' in ln and 'ts=' in ln and 'pts=' in ln:
            try:
                pts_idx = ln.index('pts=') + 4
                pts_rest = ln[pts_idx:]
                pts_str = pts_rest.split(',')[0]
                pts = int(pts_str.strip())
                if pts > 0:
                    # scan id
                    sid_idx = ln.index('scan_id=') + len('scan_id=')
                    sid_rest = ln[sid_idx:]
                    sid = int(sid_rest.split(',')[0])
                    idx = ln.index('ts=') + 3
                    rest = ln[idx:]
                    rest = rest.split(')')[0]
                    ts = float(rest)
                    return (sid, ts)
            except Exception:
                continue

    for i, ln in enumerate(lines):
        if 'Callback invoked' in ln and 'scan_id=' in ln:
            try:
                sid_idx = ln.index('scan_id=') + len('scan_id=')
                sid_rest = ln[sid_idx:]
                sid = int(sid_rest.split(')')[0].strip().strip(','))
            except Exception:
                sid = None
            # scan forward for Timestamp line
            for j in range(i + 1, min(i + 6, len(lines))):
                if 'Timestamp:' in lines[j]:
                    try:
                        parts = lines[j].split('Timestamp:')[1].strip()
                        ts_val = float(parts.split()[0])
                        return (sid, ts_val)
                    except Exception:
                        continue

    for i, ln in enumerate(lines):
        if 'Timestamp:' in ln:
            try:
                parts = ln.split('Timestamp:')[1].strip()
                ts_val = float(parts.split()[0])
            except Exception:
                continue
            for j in range(i + 1, min(i + 8, len(lines))):
                if 'Point cloud:' in lines[j]:
                    try:
                        n = int(lines[j].split('Point cloud:')[1].split()[0])
                        if n > 0:
                            # attempt to find a scan id nearby
                            sid = None
                            # search nearby lines for '(scan_id=' pattern (either before or after timestamp)
                            for k in range(max(0, i-3), min(len(lines), i+8)):
                                if 'scan_id=' in lines[k]:
                                    try:
                                        sid_idx = lines[k].index('scan_id=') + len('scan_id=')
                                        sid_rest = lines[k][sid_idx:]
                                        # strip ')' and ',' and whitespace
                                        sid = int(sid_rest.split(')')[0].split()[0].strip().strip(',').strip('()'))
                                        break
                                    except Exception:
                                        sid = None
                            return (sid, ts_val)
                    except Exception:
                        pass
    return (None, None)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    try:
        subprocess.check_call(["make", "-C", "."])
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(2)

    env = dict(**subprocess.os.environ)
    env['PELORUS_DUMP_ALL_POINTS'] = '1'

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    log("Running async_callback...")
    out_async, err_async, rc_async = run_and_collect_until("./async_callback", env=env, sentinels=[r"Callback invoked! Odometry #\d+ received \(scan_id=\d+\)"])

    log("Running sync_blocking...")
    out_sync, err_sync, rc_sync = run_and_collect_until("./sync_blocking", env=env, sentinels=[r"Point cloud: \d+ points"]) 

    out_cpp = None
    try:
        subprocess.run("make cpp_wrapper", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if subprocess.os.path.exists("build/cpp_wrapper"):
            log("Running cpp_wrapper (build/cpp_wrapper)...")
            out_cpp, err_cpp, rc_cpp = run_and_collect_until(
                "./build/cpp_wrapper", env=env,
                sentinels=[r"Point cloud: \d+ points", r"Callback invoked! Odometry #\d+ received "])
            if rc_cpp != 0:
                print(f"Error: cpp_wrapper exited with code {rc_cpp}. stdout/stderr follow:\n{out_cpp}\n{err_cpp}")
                sys.exit(1)
        else:
            print("Error: cpp_wrapper binary missing after build")
            sys.exit(2)
    except subprocess.CalledProcessError:
        print("Error: cpp_wrapper build failed")
        sys.exit(2)

    can_run_py = True
    try:
        subprocess.run([python_exe, "-c", "import pelorus"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        try:
            subprocess.run([python_exe, "-m", "pip", "install", "-e", "../../pelorus_py"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run([python_exe, "-c", "import pelorus"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print("Error: Failed to install/import pelorus Python bindings")
            sys.exit(2)

    pts_py = None
    py_returncode = None
    py_stdout = None
    py_stderr = None
    if can_run_py:
        try:
            py_stdout, py_stderr, py_returncode = run_and_collect_until(f"{python_exe} ../../pelorus_py/examples/async_callback.py", env=env, sentinels=[r"Matched odom:", r"Callback invoked! Odometry #\d+ received \(scan_id=\d+\)"])
            if py_stdout is None:
                pts_py = None
            else:
                pts_py = parse_points(py_stdout)
        except Exception:
            pts_py = None
            py_stdout = None
            py_stderr = None
            py_returncode = None


    pts_async = parse_points(out_async)
    pts_sync = parse_points(out_sync)
    pts_cpp = parse_points(out_cpp) if out_cpp is not None else None

    # pts_py may be None if the python example failed
    pts_py = pts_py if 'pts_py' in locals() else None

    if not pts_async:
        print(out_async)
        print("Error: No points parsed from async output")
        sys.exit(2)
    if not pts_sync:
        print(out_sync)
        print("Error: No points parsed from sync output")
        sys.exit(2)

    # parse registered scan info (scan id + timestamp) for each output
    sid_async, ts_async = parse_registered_scan(out_async)
    sid_sync, ts_sync = parse_registered_scan(out_sync)
    sid_cpp, ts_cpp = parse_registered_scan(out_cpp) if out_cpp is not None else (None, None)
    sid_py, ts_py = parse_registered_scan(py_stdout) if py_stdout is not None else (None, None)

    if sid_async is None or sid_sync is None or sid_async == 0 or sid_sync == 0:
        print("Error: could not identify registered scan id from async/sync outputs")
        print("--- async output start ---")
        print(out_async)
        print("--- async output end ---")
        print("--- sync output start ---")
        print(out_sync)
        print("--- sync output end ---")
        sys.exit(1)

    # Expect async and sync to report the same scan id and timestamp
    if sid_async != sid_sync:
        print(f"Error: async and sync reported different scan ids: async={sid_async} sync={sid_sync}")
        sys.exit(1)
    if ts_async is None or ts_sync is None:
        print("Error: could not parse timestamps from async/sync outputs")
        sys.exit(1)
    if abs(ts_async - ts_sync) > 1e-6:
        print(f"Timestamp mismatch between async and sync: async={ts_async} sync={ts_sync}")
        sys.exit(1)

    if py_returncode is not None and py_returncode != 0:
        print(f"Error: python example exited with code {py_returncode}. stdout/stderr follow:\n{py_stdout}\n{py_stderr}")
        sys.exit(1)

    if len(pts_async) != len(pts_sync):
        print(f"Different point counts: async={len(pts_async)} sync={len(pts_sync)}")
        sys.exit(1)

    eps = 1e-4
    for i, (a, b) in enumerate(zip(pts_async, pts_sync)):
        for k in range(4):
            if not isclose(a[k], b[k], rel_tol=1e-6, abs_tol=eps):
                print(f"Point {i} differs between async and sync at element {k}: async={a[k]} sync={b[k]}")
                sys.exit(1)

    print("SUCCESS: async and sync point clouds match exactly within tolerance")

    if pts_py is not None:
        if not pts_py:
            print("python example produced no points; failing")
            sys.exit(2)
        if len(pts_py) != len(pts_async):
            print(f"Different point counts: async={len(pts_async)} python={len(pts_py)}")
            sys.exit(1)
        for i, (a, p) in enumerate(zip(pts_async, pts_py)):
            for k in range(4):
                if not isclose(a[k], p[k], rel_tol=1e-6, abs_tol=eps):
                    print(f"Point {i} differs between async and python at element {k}: async={a[k]} python={p[k]}")
                    sys.exit(1)
        # Timestamp check: python timestamp must equal async timestamp within float tol
        if ts_async is not None and ts_py is not None:
            if abs(ts_async - ts_py) > 1e-6:
                print(f"Error: Timestamp differs: async={ts_async} python={ts_py}")
                sys.exit(1)
        else:
            print(py_stdout)
            print("Error: could not parse timestamp from python output; failing")
            sys.exit(1)

        print("SUCCESS: python example matches async/sync point clouds and timestamps")
    else:
        print("python example not run or failed")
        sys.exit(1)

    if not pts_cpp:
        print("Error: cpp_wrapper produced no points; failing")
        sys.exit(2)
    if len(pts_cpp) != len(pts_async):
        print(f" Error: Different point counts: async={len(pts_async)} cpp={len(pts_cpp)}")
        sys.exit(1)
    for i, (a, c) in enumerate(zip(pts_async, pts_cpp)):
        for k in range(4):
            if not isclose(a[k], c[k], rel_tol=1e-6, abs_tol=eps):
                print(f"Point {i} differs between async and cpp at element {k}: async={a[k]} cpp={c[k]}")
                sys.exit(1)
    # Timestamp check for cpp wrapper
    if sid_cpp is None or sid_async is None or sid_cpp != sid_async:
        print(f"Error: cpp_wrapper scan id mismatch: async={sid_async} cpp={sid_cpp}")
        sys.exit(1)
    if ts_async is not None and ts_cpp is not None:
        if abs(ts_async - ts_cpp) > 1e-6:
            print(f"Error: Timestamp differs: async={ts_async} cpp={ts_cpp}")
            sys.exit(1)

    print("SUCCESS: cpp_wrapper matches async point clouds and timestamp")

if __name__ == '__main__':
    main()
