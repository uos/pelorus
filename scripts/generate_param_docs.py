#!/usr/bin/env python3
"""
Generate parameter reference documentation from default.rl configuration file.

This script parses the default.rl file and generates structured markdown documentation
for all configuration parameters, preserving comments as descriptions.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def parse_default_rl(content: str) -> List[Dict]:
    """Parse default.rl and extract parameters with their metadata."""
    lines = content.split("\n")
    parameters = []
    current_path = []
    current_comments = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            current_comments = []
            continue

        # Collect comments
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment:
                current_comments.append(comment)
            continue

        # Section opening
        if stripped.endswith("{"):
            section_name = stripped[:-1].strip()
            current_path.append(section_name)
            current_comments = []
            continue

        # Section closing
        if stripped == "}":
            if current_path:
                current_path.pop()
            current_comments = []
            continue

        # Parameter assignment
        if "=" in stripped and not stripped.startswith("}"):
            # Extract parameter name and value
            match = re.match(r"^(\S+)\s*=\s*(.+?)(?:\s*#.*)?$", stripped)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).rstrip()

                # Build full path
                full_path = ".".join(current_path + [param_name])

                # Determine type from value
                param_type = infer_type(param_value)

                # Check if recommended
                is_recommended = any("@recommended" in c for c in current_comments)

                # Filter out @recommended tags from comments
                filtered_comments = [
                    c for c in current_comments if "@recommended" not in c
                ]

                parameters.append(
                    {
                        "path": full_path,
                        "name": param_name,
                        "value": param_value,
                        "type": param_type,
                        "comments": filtered_comments,
                        "recommended": is_recommended,
                        "depth": len(current_path),
                    }
                )

                current_comments = []

    return parameters


def infer_type(value: str) -> str:
    """Infer the type of a parameter from its value."""
    value = value.strip()

    # Check for string with quotes
    if value.startswith('"') and value.endswith('"'):
        return "string"

    # Check for array
    if value.startswith("["):
        return "array"

    # Check for boolean
    if value.lower() in ["true", "false"]:
        return "boolean"

    # Check for number with units
    if re.match(r"^\d+(\.\d+)?(m|cm|mm|s|ms)$", value):
        return "number (with unit)"

    # Check for plain number
    if re.match(r"^\d+(\.\d+)?$", value):
        return "number"

    # Check for range
    if ".." in value:
        return "range"

    # Check for enum-like values (capitalized)
    if re.match(r"^[A-Z][a-zA-Z]+$", value):
        return "enum"

    return "string"


def group_parameters_by_section(parameters: List[Dict]) -> Dict[str, List[Dict]]:
    """Group parameters by their top-level section."""
    groups = {}
    for param in parameters:
        section = param["path"].split(".")[0]
        if section not in groups:
            groups[section] = []
        groups[section].append(param)
    return groups


def generate_markdown(parameters: List[Dict], group_name: str) -> str:
    """Generate markdown documentation for a group of parameters."""
    md = f"## {group_name.title()}\n\n"

    # Find parameters at different depths
    for param in parameters:
        indent = "  " * (param["depth"] - 1)

        # Header
        header_level = min(param["depth"] + 2, 6)
        md += f"{'#' * header_level} `{param['path']}`\n\n"

        if param["recommended"]:
            md += f"**Recommended** parameter\n\n"

        # Description from comments
        if param["comments"]:
            description = " ".join(param["comments"])
            md += f"{description}\n\n"

        # Metadata table
        md += f"| Property | Value |\n"
        md += f"|----------|-------|\n"
        md += f"| Type | `{param['type']}` |\n"
        md += f"| Default | `{param['value']}` |\n"
        md += f"| Path | `{param['path']}` |\n"
        md += "\n"

        # Examples based on type
        if param["type"] == "number (with unit)":
            md += '!!! tip "Units"\n'
            md += "    Supports units: `m`, `cm`, `mm`, `s`, `ms`\n\n"
        elif param["type"] == "enum":
            md += '!!! info "Enum Value"\n'
            md += f"    This parameter accepts specific enum values.\n\n"

        md += "---\n\n"

    return md


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_param_docs.py <default.rl_path>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read default.rl
    with open(input_path, "r") as f:
        content = f.read()

    # Parse parameters
    parameters = parse_default_rl(content)

    # Generate header
    markdown = """---
icon: lucide/settings
---

# Parameter Reference

This document provides a comprehensive reference for all Pelorus configuration parameters. The parameters are extracted from the `default.rl` configuration file, which serves as the single source of truth.

For a guide on how to configure Pelorus, see the [Configuration Guide](../configuration.md).

!!! tip "Units"
    Pelorus supports human-readable units in configuration files:
    
    - **Distance**: `m` (meters), `cm` (centimeters), `mm` (millimeters)
    - **Time**: `s` (seconds), `ms` (milliseconds)
    
    Example: `resolution = 50cm` is equivalent to `resolution = 0.5`

"""

    # Group and generate
    groups = group_parameters_by_section(parameters)

    # Order sections logically
    section_order = [
        "registration",
        "preprocessing",
        "lidar",
        "imu",
        "wheel",
        "output",
        "parallelism",
    ]

    for section in section_order:
        if section in groups:
            markdown += generate_markdown(groups[section], section)

    # Output
    print(markdown)


if __name__ == "__main__":
    main()
