# Configuration

Pelorus is configured via [ratslang](https://codeberg.org/stelzo/ratslang). The binary ships the default and minimal configuration, which can be used as a starting point.

You can generate a configuration file template with the CLI or the ROS 2 node:

```sh
# prepend 'ros2 run pelorus' if you only have the ROS 2 package installed 
pelorus config generate recommended > pelorus.rl
```

This will create a minimal configuration file `pelorus.rl` in the current directory that should suffice for most applications. You will need the path to this file to run Pelorus.

For editing your configuration, we recommend using the Ratslang VSCode Extension or the neovim and helix plugins.

After editing, you can validate your configuration against the Pelorus schema using:

```sh
pelorus config validate pelorus.rl
```

If nothing is printed, your file is valid and ready to use.

# TODO maybe generate the default config here from pelorus cli in the ci release

The generated file is the single source of truth and self-documenting. It undergoes type and usage checks at compile time to really make invalid configurations impossible.

If you do not manually validate your edited configuration, Pelorus will do it at startup. If anything goes wrong, it will quit with an error message or give you warnings.

## Experimental multi-resolution local map

Enable the CPU registration prototype with:

```rl
registration.map.multiresolution = true
```

The default is `false`, which preserves the fixed-resolution map. The prototype
selects CPU registration even in a binary with GPU features. Preprocessing still
honors `preprocessing.downsample.backend`; use `CPU` there if you also want CPU
downsampling. The experimental `kdtree` feature cannot be combined with this map.

With `registration.map.resolution = 50cm`, the four voxel sizes are 25 cm,
50 cm, 1 m, and 2 m. The configured capacity is a **total cell budget**, shared
equally across the levels, not a capacity per level. It is not an equal-byte
memory budget: cached neighborhood fits add storage per cell. Query support
includes the cell and its 26 neighbors at the selected level.

Plane selection checks point support, surface thickness, line degeneracy, and
the observed patch's tangent extent. Sparse or line-like support may search a
coarser level; a sufficiently supported nonplanar neighborhood stops expansion
so surrounding points cannot dilute a geometric conflict. A larger support must agree with the finer
surface and improve a geometric normal-quality score. Each scan point supplies
at most one plane residual. Normal agreement also depends on the two fits'
quality scores: well-constrained normals tolerate less angular disagreement.
Cached neighborhood fits are invalidated after every
map update and eviction. Scan downsampling is unchanged: the finest map cannot
recover details discarded during preprocessing.

This prototype does not implement calibrated plane uncertainty, loop closure,
or GPU map updates. It is opt-in pending indoor/transition validation and the
outdoor accuracy regression gate. The accompanying experiment record is
[Multi-resolution local-map evaluation](multiresolution-roundabout02.md).

To evaluate two complete trajectories with Sounding, keeping the reference
second as required by its CLI:

```sh
python3 scripts/compare-localmap-trajectories.py \
  baseline.tum multiresolution.tum \
  "$(marina resolve helipr/roundabout02:ouster)/LiDAR_GT/Ouster_gt.txt" \
  --output comparison.json
```

The script requires identical scan timestamps and returns success only if all
four translation/rotation APE/RPE RMSE values are non-increasing. Use a complete
baseline for this check; equal-length partial trajectories do not establish
full-sequence accuracy. Sounding's default alignment and a one-second RPE delta
are used for both trajectories.
