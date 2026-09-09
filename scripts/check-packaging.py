#!/usr/bin/env python3
"""
Validates that packaging files (PKGBUILD, deb metadata, rpm metadata, pyproject.toml)
have version constraints for minot/minot-cli that match the workspace Cargo.toml dependencies.

Usage:
    python3 scripts/check-packaging.py          # check only (exits 1 on mismatch)
    python3 scripts/check-packaging.py --fix    # automatically fix mismatches
"""

import sys
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        print("ERROR: Python 3.11+ (tomllib) or tomli is required.", file=sys.stderr)
        sys.exit(1)


def get_workspace_info(root: Path):
    cargo_path = root / "Cargo.toml"
    with open(cargo_path, "rb") as f:
        cargo = tomllib.load(f)

    workspace_ver = cargo.get("workspace", {}).get("package", {}).get("version")
    if not workspace_ver:
        raise ValueError("Could not find workspace.package.version in Cargo.toml")

    ws_deps = cargo.get("workspace", {}).get("dependencies", {})
    mt_ver = None
    for k in ["mt_pubsub", "mt_rat", "mt_service", "minot"]:
        val = ws_deps.get(k)
        if isinstance(val, str):
            mt_ver = val
            break
        elif isinstance(val, dict) and "version" in val:
            mt_ver = val["version"]
            break

    if not mt_ver:
        raise ValueError("Could not find mt_pubsub / minot dependency version in Cargo.toml workspace.dependencies")

    # Parse semver major/minor
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", mt_ver)
    if not match:
        raise ValueError(f"Invalid minot version format: {mt_ver}")

    major = int(match.group(1))
    minor = int(match.group(2))

    if major == 0:
        min_ver = f"0.{minor}.0"
        next_ver = f"0.{minor + 1}.0"
    else:
        min_ver = f"{major}.0.0"
        next_ver = f"{major + 1}.0.0"

    return {
        "workspace_ver": workspace_ver,
        "minot_ver": mt_ver,
        "min_ver": min_ver,
        "next_ver": next_ver,
    }


def check_pkgbuild(root: Path, info: dict, fix: bool) -> list[str]:
    errors = []
    pkgbuild_path = root / "packaging" / "arch-bin" / "PKGBUILD"
    if not pkgbuild_path.exists():
        return [f"{pkgbuild_path} does not exist"]

    content = pkgbuild_path.read_text(encoding="utf-8")
    new_content = content

    expected_minot = f"'minot>={info['min_ver']}' 'minot<{info['next_ver']}'"

    # Check pkgver
    pkgver_match = re.search(r"^pkgver=(.+)$", content, re.MULTILINE)
    if pkgver_match:
        current_pkgver = pkgver_match.group(1).strip()
        if current_pkgver != info["workspace_ver"]:
            msg = (
                f"PKGBUILD pkgver mismatch: found '{current_pkgver}', "
                f"expected workspace version '{info['workspace_ver']}'"
            )
            errors.append(msg)
            if fix:
                new_content = re.sub(
                    r"^pkgver=.+$",
                    f"pkgver={info['workspace_ver']}",
                    new_content,
                    flags=re.MULTILINE,
                )

    # Check depends for minot
    # Pattern looks for minot constraints in depends=(...)
    minot_match = re.search(r"'minot[><=][^']*'(?:\s+'minot[><=][^']*')?", content)
    if minot_match:
        current_minot = minot_match.group(0)
        if current_minot != expected_minot:
            msg = (
                f"PKGBUILD minot dependency mismatch:\n"
                f"  found:    {current_minot}\n"
                f"  expected: {expected_minot}"
            )
            errors.append(msg)
            if fix:
                new_content = new_content.replace(current_minot, expected_minot)
    else:
        errors.append(f"Could not locate minot dependency pattern in {pkgbuild_path}")

    if fix and new_content != content:
        pkgbuild_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {pkgbuild_path.relative_to(root)}")

    return errors


def check_pelorus_cli_cargo(root: Path, info: dict, fix: bool) -> list[str]:
    errors = []
    cli_cargo_path = root / "pelorus_cli" / "Cargo.toml"
    if not cli_cargo_path.exists():
        return [f"{cli_cargo_path} does not exist"]

    content = cli_cargo_path.read_text(encoding="utf-8")
    new_content = content

    expected_deb = f'minot (>= {info["min_ver"]}), minot (<< {info["next_ver"]}~)'
    expected_rpm = f'>= {info["min_ver"]}, < {info["next_ver"]}'

    # Check deb depends
    deb_match = re.search(r'\[package\.metadata\.deb\][\s\S]*?depends\s*=\s*"([^"]+)"', content)
    if deb_match:
        current_deb_all = deb_match.group(1)
        # Extract minot portion
        minot_deb_match = re.search(r'minot\s*\([^)]+\)(?:,\s*minot\s*\([^)]+\))?', current_deb_all)
        if minot_deb_match:
            current_deb_minot = minot_deb_match.group(0)
            if current_deb_minot != expected_deb:
                msg = (
                    f"pelorus_cli/Cargo.toml [package.metadata.deb] mismatch:\n"
                    f"  found:    {current_deb_minot}\n"
                    f"  expected: {expected_deb}"
                )
                errors.append(msg)
                if fix:
                    new_content = new_content.replace(current_deb_minot, expected_deb)
    else:
        errors.append("Could not find [package.metadata.deb].depends in pelorus_cli/Cargo.toml")

    # Check rpm requires
    rpm_match = re.search(r'\[package\.metadata\.generate-rpm\.requires\][\s\S]*?minot\s*=\s*"([^"]+)"', content)
    if rpm_match:
        current_rpm = rpm_match.group(1)
        if current_rpm != expected_rpm:
            msg = (
                f"pelorus_cli/Cargo.toml [package.metadata.generate-rpm.requires] mismatch:\n"
                f"  found:    minot = \"{current_rpm}\"\n"
                f"  expected: minot = \"{expected_rpm}\""
            )
            errors.append(msg)
            if fix:
                old_line = f'minot = "{current_rpm}"'
                new_line = f'minot = "{expected_rpm}"'
                new_content = new_content.replace(old_line, new_line)
    else:
        errors.append("Could not find [package.metadata.generate-rpm.requires].minot in pelorus_cli/Cargo.toml")

    if fix and new_content != content:
        cli_cargo_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {cli_cargo_path.relative_to(root)}")

    return errors


def check_pyproject(root: Path, info: dict, fix: bool) -> list[str]:
    errors = []
    pyproject_path = root / "pelorus_cli" / "pyproject.toml"
    if not pyproject_path.exists():
        return [f"{pyproject_path} does not exist"]

    content = pyproject_path.read_text(encoding="utf-8")
    new_content = content

    expected_py_dep = f"minot-cli>={info['min_ver']},<{info['next_ver']}"

    match = re.search(r'minot-cli[><=0-9.,]+', content)
    if match:
        current_py_dep = match.group(0)
        if current_py_dep != expected_py_dep:
            msg = (
                f"pelorus_cli/pyproject.toml dependencies mismatch:\n"
                f"  found:    {current_py_dep}\n"
                f"  expected: {expected_py_dep}"
            )
            errors.append(msg)
            if fix:
                new_content = new_content.replace(current_py_dep, expected_py_dep)
    else:
        errors.append("Could not find minot-cli dependency in pelorus_cli/pyproject.toml")

    if fix and new_content != content:
        pyproject_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {pyproject_path.relative_to(root)}")

    return errors


def main():
    root = Path(__file__).resolve().parent.parent
    fix = "--fix" in sys.argv

    try:
        info = get_workspace_info(root)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Workspace version: {info['workspace_ver']}")
    print(f"Minot dependency:  {info['minot_ver']} (expected range: >={info['min_ver']}, <{info['next_ver']})")
    print("-" * 60)

    all_errors = []
    all_errors.extend(check_pkgbuild(root, info, fix))
    all_errors.extend(check_pelorus_cli_cargo(root, info, fix))
    all_errors.extend(check_pyproject(root, info, fix))

    if fix:
        print("-" * 60)
        print("Re-checking after applying fixes...")
        all_errors = []
        all_errors.extend(check_pkgbuild(root, info, fix=False))
        all_errors.extend(check_pelorus_cli_cargo(root, info, fix=False))
        all_errors.extend(check_pyproject(root, info, fix=False))

    if all_errors:
        print("\nPackaging version validation FAILED:")
        for err in all_errors:
            print(f"\n[!] {err}")
        print("\nRun `python3 scripts/check-packaging.py --fix` to automatically sync versions.")
        sys.exit(1)

    print("\nAll packaging minot version constraints are valid and in sync!")
    sys.exit(0)


if __name__ == "__main__":
    main()
