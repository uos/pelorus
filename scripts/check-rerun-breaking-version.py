#!/usr/bin/env python3
"""Fail when Pelorus is not on rerun's latest stable breaking-version line."""

from __future__ import annotations

import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


CRATE = "rerun"
CRATES_IO_URL = f"https://crates.io/api/v1/crates/{CRATE}"
VERSION_REQUIREMENT = re.compile(
    r"\s*\^?(?P<major>0|[1-9]\d*)"
    r"(?:\.(?P<minor>0|[1-9]\d*))?"
    r"(?:\.(?P<patch>0|[1-9]\d*))?\s*"
)
STABLE_VERSION = re.compile(
    r"(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
)


def parse_version(value: str, pattern: re.Pattern[str]) -> tuple[int, int, int]:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported version format: {value!r}")
    return tuple(int(match.group(part) or 0) for part in ("major", "minor", "patch"))


def breaking_line(version: tuple[int, int, int]) -> tuple[int, ...]:
    major, minor, patch = version
    if major:
        return (major,)
    if minor:
        return (major, minor)
    return (major, minor, patch)


def format_line(line: tuple[int, ...]) -> str:
    return ".".join(str(component) for component in line)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    with (repo_root / "Cargo.toml").open("rb") as cargo_toml:
        manifest = tomllib.load(cargo_toml)

    dependency = manifest["workspace"]["dependencies"][CRATE]
    requirement = dependency if isinstance(dependency, str) else dependency["version"]
    declared = parse_version(requirement, VERSION_REQUIREMENT)

    request = urllib.request.Request(
        CRATES_IO_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "pelorus-ci-version-check (https://codeberg.org/stelzo/pelorus)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: could not query {CRATES_IO_URL}: {error}", file=sys.stderr)
        return 2

    latest_text = metadata["crate"]["max_stable_version"]
    latest = parse_version(latest_text, STABLE_VERSION)
    declared_line = breaking_line(declared)
    latest_line = breaking_line(latest)

    print(f"Pelorus rerun requirement: {requirement} (breaking line {format_line(declared_line)})")
    print(f"Latest stable rerun:       {latest_text} (breaking line {format_line(latest_line)})")

    if declared_line != latest_line:
        print(
            "::error title=New rerun breaking release available::"
            f"Pelorus requires rerun {requirement}, but the latest stable release is "
            f"{latest_text}. Update rerun to the {format_line(latest_line)} line.",
            file=sys.stderr,
        )
        return 1

    print("Pelorus uses rerun's latest stable breaking-version line.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as error:
        print(f"error: invalid rerun version metadata: {error}", file=sys.stderr)
        raise SystemExit(2) from error
