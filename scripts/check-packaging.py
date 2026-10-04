#!/usr/bin/env python3
"""
Validates that packaging files (PKGBUILD, deb metadata, rpm metadata, pyproject.toml)
have version constraints for minot/minot-cli and rerun-sdk that match the workspace
Cargo.toml dependencies.

Constraints derived from Cargo.toml can be repaired with --fix. The versions the
documentation tells a reader to install cannot: a release has to exist upstream
before the docs can name it, and this script has no way to know that. Those are
reported only, which is why --docs-only exists as a pre-flight for a release.

Usage:
    python3 scripts/check-packaging.py             # check only (exits 1 on mismatch)
    python3 scripts/check-packaging.py --fix       # repair what can be derived
    python3 scripts/check-packaging.py --docs-only # only the checks --fix cannot repair
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

    min_ver, next_ver = compat_range(mt_ver, "minot")

    # The Rust crate `rerun` and the Python package `rerun-sdk` are cut from the
    # same release and share version numbers, so the optional dependency that
    # installs the viewer has to track the crate the CLI is built against.
    rr_ver = None
    val = ws_deps.get("rerun")
    if isinstance(val, str):
        rr_ver = val
    elif isinstance(val, dict) and "version" in val:
        rr_ver = val["version"]

    if not rr_ver:
        raise ValueError("Could not find rerun dependency version in Cargo.toml workspace.dependencies")

    rerun_min_ver, rerun_next_ver = compat_range(rr_ver, "rerun")

    # The docs tell people to `cargo install zenohd@...` for the ROS 2 over
    # Zenoh setup, which has to be the same release line Pelorus links against.
    zenoh_ver = None
    val = ws_deps.get("zenoh-buffers")
    if isinstance(val, str):
        zenoh_ver = val
    elif isinstance(val, dict) and "version" in val:
        zenoh_ver = val["version"]

    if not zenoh_ver:
        raise ValueError("Could not find zenoh-buffers dependency version in Cargo.toml workspace.dependencies")

    return {
        "workspace_ver": workspace_ver,
        "minot_ver": mt_ver,
        "min_ver": min_ver,
        "next_ver": next_ver,
        "rerun_ver": rr_ver,
        "rerun_min_ver": rerun_min_ver,
        "rerun_next_ver": rerun_next_ver,
        "zenoh_ver": zenoh_ver,
        "cuda_variants": cuda_variants(root),
    }


def cuda_variants(root: Path) -> dict[str, set[str]]:
    """CUDA variants the release workflow actually builds, as "12-8" and so on.

    The GLIM plugin is kept apart because it is offered for fewer of them: the
    plugin is a dependency declaration on a libpelorus variant, and GLIM upstream
    supports a narrower set of CUDA versions than libpelorus is built for.

    Commented-out matrix rows are skipped, so a variant that is on hold does not
    have to be documented.
    """
    release = root / ".forgejo" / "workflows" / "release.yml"
    if not release.exists():
        raise ValueError(f"{release} does not exist")

    text = release.read_text(encoding="utf-8")

    general = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"^(?:-\s*)?cuda:\s*([0-9]+-[0-9]+)\s*$", stripped)
        if match:
            general.add(match.group(1))

    glim = set(re.findall(r'glim_odometry-cuda([0-9]+-[0-9]+)"', text))

    if not general:
        raise ValueError(f"Could not find any cuda matrix entries in {release}")
    if not glim:
        raise ValueError(f"Could not find any glim cuda package names in {release}")
    return {"general": general, "glim": glim}


def major_minor(version: str) -> str:
    return ".".join(version.lstrip("v").split(".")[:2])


def compat_range(version: str, name: str) -> tuple[str, str]:
    """Semver-compatible range for a dependency, as (minimum, exclusive upper).

    Below 1.0 a minor bump is breaking, so the range is pinned to one minor line.
    """
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        raise ValueError(f"Invalid {name} version format: {version}")

    major = int(match.group(1))
    minor = int(match.group(2))

    if major == 0:
        return f"0.{minor}.0", f"0.{minor + 1}.0"
    return f"{major}.0.0", f"{major + 1}.0.0"


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


def check_rerun_pyproject(root: Path, info: dict, fix: bool) -> list[str]:
    """The `rerun` optional dependency of pelorus-cli, which installs the viewer.

    This is the one that goes stale unnoticed, because nothing fails to build
    when it drifts. A user ends up with a viewer that cannot talk to the
    recording stream the CLI produces.
    """
    errors = []
    pyproject_path = root / "pelorus_cli" / "pyproject.toml"
    if not pyproject_path.exists():
        return [f"{pyproject_path} does not exist"]

    content = pyproject_path.read_text(encoding="utf-8")
    new_content = content

    expected = f"rerun-sdk>={info['rerun_min_ver']},<{info['rerun_next_ver']}"

    match = re.search(r"rerun-sdk[><=0-9.,]+", content)
    if match:
        current = match.group(0)
        if current != expected:
            msg = (
                f"pelorus_cli/pyproject.toml rerun extra mismatch:\n"
                f"  found:    {current}\n"
                f"  expected: {expected}  (workspace rerun crate is {info['rerun_ver']})"
            )
            errors.append(msg)
            if fix:
                new_content = new_content.replace(current, expected)
    else:
        errors.append("Could not find rerun-sdk dependency in pelorus_cli/pyproject.toml")

    if fix and new_content != content:
        pyproject_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {pyproject_path.relative_to(root)}")

    return errors


def check_docs(root: Path, info: dict, fix: bool) -> list[str]:
    """Versions the documentation tells the reader to install.

    A page that pins a version is making a claim about what works with this
    checkout, and nothing fails to build when that claim goes stale. The reader
    is the one who finds out. None of these are rewritten automatically: a
    version has to exist upstream before the docs can name it, and this script
    cannot know that.
    """
    errors = []
    docs_dir = root / "docs" / "docs"
    if not docs_dir.exists():
        return [f"{docs_dir} does not exist"]

    pages = {path: path.read_text(encoding="utf-8") for path in sorted(docs_dir.rglob("*.md"))}

    def where(pattern: str):
        """Every page, line number and match for a pattern, so a stale version is
        reported at the line the reader would have copied."""
        for path, text in pages.items():
            for number, line in enumerate(text.splitlines(), start=1):
                for match in re.finditer(pattern, line):
                    yield f"{path.relative_to(root)}:{number}", match

    # `cargo install rerun-cli@X.Y.Z`, which has to be the release line the
    # CLI was compiled against. The docs install the viewer through the
    # `pelorus-cli[rerun]` extra, which check_rerun_pyproject covers, so a
    # rerun-cli pin is optional here.
    rerun_line = major_minor(info["rerun_ver"])
    for rel, match in where(r"rerun-cli@([0-9]+\.[0-9]+\.[0-9]+)"):
        if major_minor(match.group(1)) != rerun_line:
            errors.append(
                f"{rel} pins rerun-cli@{match.group(1)}, but the workspace rerun "
                f"crate is {info['rerun_ver']}. Name a published {rerun_line}.x release."
            )

    # `cargo install ... minot --tag vX.Y.Z`, which has to be the release the
    # workspace depends on. The docs may install minot without a tag, so a pin
    # is optional here.
    #
    # minot used to publish a separate `-hiroz` tag line for the ROS 2 over
    # Zenoh build. That line stopped at v0.9.1-hiroz once hiroz support became a
    # feature of the mainline, so a suffixed tag in the docs now means a stale
    # command and not a deliberate second line.
    minot_line = major_minor(info["minot_ver"])
    for rel, match in where(r"minot\s+--tag\s+v([0-9]+\.[0-9]+\.[0-9]+)(\S*)"):
        version, suffix = match.group(1), match.group(2)
        if suffix:
            errors.append(
                f"{rel} pins the tag v{version}{suffix}. The suffixed minot tags are "
                f"retired, and hiroz is a feature of the mainline now, so name a "
                f"{minot_line}.x tag and select the feature instead."
            )
        elif major_minor(version) != minot_line:
            errors.append(
                f"{rel} installs minot from tag v{version}, but the workspace depends on "
                f"minot {info['minot_ver']}. Name a {minot_line}.x tag."
            )
    # `cargo install zenohd@X.Y.Z`, which has to match what Pelorus links.
    zenoh_line = major_minor(info["zenoh_ver"])
    for rel, match in where(r"zenohd@([0-9]+\.[0-9]+\.[0-9]+)"):
        if major_minor(match.group(1)) != zenoh_line:
            errors.append(
                f"{rel} pins zenohd@{match.group(1)}, but the workspace links zenoh "
                f"{info['zenoh_ver']}. Name a {zenoh_line}.x release."
            )

    # Accelerated package variants, per family. Every variant the release builds
    # has to be offered, and a variant that is not built must not be. The GLIM
    # plugin is counted on its own, because it is offered for fewer of them.
    documented = {"general": set(), "glim": set()}
    for _, match in where(r"(glim[-_]odometry-)?cuda([0-9]+-[0-9]+)"):
        family = "glim" if match.group(1) else "general"
        documented[family].add(match.group(2))

    labels = {
        "general": "the library, CLI, ROS node and wheels",
        "glim": "the GLIM plugin",
    }
    for family, built in info["cuda_variants"].items():
        missing = sorted(built - documented[family])
        if missing:
            errors.append(
                f"The release builds CUDA variants of {labels[family]} that the docs "
                f"never mention: {', '.join(missing)}. Add them to docs/docs/install.md."
            )

        unbuilt = sorted(documented[family] - built)
        if unbuilt:
            errors.append(
                f"The docs offer CUDA variants of {labels[family]} that the release does "
                f"not build: {', '.join(unbuilt)}. Remove them from docs/docs/install.md."
            )

    return errors


def main():
    root = Path(__file__).resolve().parent.parent
    fix = "--fix" in sys.argv
    docs_only = "--docs-only" in sys.argv

    if fix and docs_only:
        print("ERROR: --fix and --docs-only are mutually exclusive, because the", file=sys.stderr)
        print("       documentation checks are the ones --fix cannot repair.", file=sys.stderr)
        sys.exit(2)

    try:
        info = get_workspace_info(root)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Workspace version: {info['workspace_ver']}")
    print(f"Minot dependency:  {info['minot_ver']} (expected range: >={info['min_ver']}, <{info['next_ver']})")
    print(f"Rerun dependency:  {info['rerun_ver']} (expected range: >={info['rerun_min_ver']}, <{info['rerun_next_ver']})")
    print("-" * 60)

    if docs_only:
        checks = [check_docs]
    else:
        checks = [
            check_pkgbuild,
            check_pelorus_cli_cargo,
            check_pyproject,
            check_rerun_pyproject,
            check_docs,
        ]

    all_errors = []
    for check in checks:
        all_errors.extend(check(root, info, fix))

    if fix:
        print("-" * 60)
        print("Re-checking after applying fixes...")
        all_errors = []
        for check in checks:
            all_errors.extend(check(root, info, fix=False))

    if all_errors:
        print("\nPackaging version validation FAILED:")
        for err in all_errors:
            print(f"\n[!] {err}")
        if docs_only:
            print("\nThese name versions to install, so fix them by hand. Each has to exist")
            print("upstream before the docs can point at it.")
        else:
            print("\nRun `python3 scripts/check-packaging.py --fix` to sync what can be derived.")
            print("Anything above naming a version to install has to be fixed by hand.")
        sys.exit(1)

    if docs_only:
        print("\nEvery version the documentation tells a reader to install is current!")
        sys.exit(0)

    print("\nAll packaging minot and rerun version constraints are valid and in sync!")
    sys.exit(0)


if __name__ == "__main__":
    main()
