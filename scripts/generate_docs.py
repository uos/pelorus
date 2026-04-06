#!/usr/bin/env python3
"""
Generate all remaining Pelorus documentation files.
This script creates placeholder documentation files with frontmatter and structure.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent / "docs" / "docs"

# Documentation files to create
DOCS = {
    "integration/c-library.md": {
        "title": "C/C++ Library",
        "icon": "lucide/code",
        "template": "Adapt from pelorus_c/README.md with API reference, CMake integration, PCL/Eigen helpers",
    },
    "integration/python.md": {
        "title": "Python Package",
        "icon": "lucide/python",
        "template": "Adapt from pelorus_py/README.md with NumPy integration, async API, MCAP processing",
    },
    "integration/glim.md": {
        "title": "GLIM Plugin",
        "icon": "lucide/puzzle",
        "template": "GLIM integration guide, installation, configuration, comparison with standalone",
    },
    "integration/cli.md": {
        "title": "Standalone CLI",
        "icon": "lucide/terminal",
        "template": "CLI commands reference, offline/online modes, config generation, evaluation",
    },
    "configuration.md": {
        "title": "Configuration Guide",
        "icon": "lucide/settings",
        "template": "MAJOR EXPANSION: Ratslang basics, workflow, all parameter categories with examples",
    },
    "config/sensor-setup.md": {
        "title": "Sensor Setup",
        "icon": "lucide/cpu",
        "template": "Supported LiDARs, IMU integration, extrinsic calibration, TF frames, multi-sensor",
    },
    "config/performance-tuning.md": {
        "title": "Performance Tuning",
        "icon": "lucide/gauge",
        "template": "CPU vs GPU, thread optimization, resolution tuning, platform-specific (Jetson, desktop)",
    },
    "tutorials/basic-setup.md": {
        "title": "Basic Setup Tutorial",
        "icon": "lucide/book-open",
        "template": "Step-by-step from zero to running, test data, screenshots at each step",
    },
    "tutorials/bagfiles.md": {
        "title": "Working with Bagfiles",
        "icon": "lucide/folder",
        "template": "MCAP processing, batch processing, evaluation mode, evo integration",
    },
    "tutorials/multi-sensor.md": {
        "title": "Multi-Sensor Setup",
        "icon": "lucide/layout-grid",
        "template": "Dual LiDAR, LiDAR+Wheel, kinematic models (Ackermann, Holonomic2D)",
    },
    "tutorials/gpu-acceleration.md": {
        "title": "GPU Acceleration",
        "icon": "lucide/zap",
        "template": "CUDA setup, Metal setup, benchmarks, troubleshooting GPU issues",
    },
    "tutorials/visualization.md": {
        "title": "Real-time Visualization",
        "icon": "lucide/eye",
        "template": "Rerun setup, RViz2 config, remote visualization, custom workflows",
    },
    "advanced/algorithm-overview.md": {
        "title": "Algorithm Overview",
        "icon": "lucide/activity",
        "template": "IEKF explanation, registration, IMU integration, map representation, reference to paper",
    },
    "advanced/custom-integrations.md": {
        "title": "Custom Integrations",
        "icon": "lucide/extension",
        "template": "Building from source, feature flags, custom sensor drivers, extending Pelorus",
    },
    "advanced/troubleshooting.md": {
        "title": "Troubleshooting",
        "icon": "lucide/wrench",
        "template": "Organized by category: installation, runtime, performance, configuration issues",
    },
    "advanced/performance-optimization.md": {
        "title": "Performance Optimization",
        "icon": "lucide/trending-up",
        "template": "Profiling tools, bottleneck identification, optimization strategies, case studies",
    },
    "api/c-api.md": {
        "title": "C API Reference",
        "icon": "lucide/file-code",
        "template": "Function reference, data structures, memory management, PCL/Eigen helpers",
    },
    "api/python-api.md": {
        "title": "Python API Reference",
        "icon": "lucide/file-text",
        "template": "Handle class, methods, return types, exceptions, type hints",
    },
    "api/config-schema.md": {
        "title": "Configuration Schema",
        "icon": "lucide/file-json",
        "template": "Generated from default.rl, parameter tree, types, validation rules",
    },
}


def create_doc_file(filepath: Path, title: str, icon: str, template: str):
    """Create a documentation file with frontmatter."""

    # Check if file already has content
    if filepath.exists():
        with open(filepath, "r") as f:
            content = f.read()
            if len(content) > 100:  # Has substantial content
                print(f"Skipping {filepath} (already has content)")
                return

    # Create content
    content = f"""---
icon: {icon}
---

# {title}

<!-- 
TODO: {template}

This file needs to be completed with:
{chr(10).join(f"- {line}" for line in template.split(", "))}
-->

## Overview

[Content to be added]

## Key Concepts

[Content to be added]

## Examples

[Content to be added]

## Next Steps

- [Related Guide](#)
- [API Reference](#)

---

<!-- 
IMAGE/GIF PLACEHOLDERS:
- Add relevant screenshots or diagrams here
- Use consistent image sizing
- Provide alt text for accessibility
-->
"""

    # Write file
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

    print(f"Created {filepath}")


def main():
    """Generate all documentation files."""
    print("Generating Pelorus documentation files...")
    print(f"Base directory: {BASE_DIR}")
    print()

    for filepath, metadata in DOCS.items():
        full_path = BASE_DIR / filepath
        create_doc_file(
            full_path, metadata["title"], metadata["icon"], metadata["template"]
        )

    print()
    print("Documentation generation complete!")
    print()
    print("Next steps:")
    print("1. Review and expand configuration.md (major expansion needed)")
    print("2. Complete integration guides (c-library.md, python.md, glim.md, cli.md)")
    print("3. Add tutorial content")
    print("4. Fill in API reference details")
    print("5. Add screenshots and GIFs where marked")


if __name__ == "__main__":
    main()
