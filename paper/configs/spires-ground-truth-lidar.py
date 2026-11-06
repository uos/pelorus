#!/usr/bin/env python3
"""Convert Oxford Spires base-frame TUM ground truth to LiDAR-frame poses.

T_world_lidar = T_world_base * T_base_lidar.
Published calibration: translation [0, 0, 0.124], quaternion xyzw [0, 0, 1, 0].
https://github.com/ori-drs/oxford_spires_dataset/blob/main/configs/sensor.yaml
"""

import argparse
import math
from pathlib import Path


def convert(line):
    fields = line.split()
    values = list(map(float, fields))
    if len(values) != 8 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected eight finite TUM values")
    _, px, py, pz, x, y, z, w = values
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError("Invalid zero quaternion")
    x, y, z, w = (v / norm for v in (x, y, z, w))
    # Rotate the base-frame lever arm into world coordinates, then compose
    # orientation with a 180-degree rotation about the base z axis.
    result = (
        px + 0.124 * 2 * (x*z + w*y),
        py + 0.124 * 2 * (y*z - w*x),
        pz + 0.124 * (1 - 2 * (x*x + y*y)),
        y, -x, w, -z,
    )
    return fields[0] + " " + " ".join(format(v, ".17g") for v in result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    converted = [convert(line) for line in args.source.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    with args.output.open("x") as output:
        output.write("\n".join(converted) + "\n")


if __name__ == "__main__":
    main()
