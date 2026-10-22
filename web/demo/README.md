# Demo data

Each demo keeps the first seconds of a recording's lidar and IMU topics,
removes point fields Pelorus does not read and stores the clouds compressed
with cloudini at 1 mm precision. Create them with `web/make-demo.sh`.

`keble_college_02_30s.mcap` is an excerpt of `keble_college_02` from the
[Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)
by the Oxford Robotics Institute, licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The
excerpt is shared under the same license. Please cite *The Oxford Spires
Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and
Radiance Field Methods* (IJRR 2025) when you use it.

`grand_tour_sbb1_30s.mcap`, `grand_tour_eth1_30s.mcap` and
`grand_tour_snow1_30s.mcap` are excerpts of the sbb1, eth1 and snow1 missions of
the
[GrandTour dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset)
by the Robotic Systems Lab at ETH Zurich, licensed under the MIT license. Please
cite *Boxi: Design Decisions in the Context of Algorithmic Performance for
Robotics* (RSS 2025) when you use them.

`helipr_town02_20s.mcap` and `helipr_bridge02_20s.mcap` are excerpts of the
Town02 and Bridge02 Ouster sequences of the
[HeLiPR dataset](https://sites.google.com/view/heliprdataset) by Seoul National
University. They keep every third point of the first 20 seconds. The dataset
does not state a license, so confirm the terms with its authors
(helipr.dataset@gmail.com) before hosting these files. Please cite *HeLiPR:
Heterogeneous LiDAR Dataset for inter-LiDAR Place Recognition under
Spatiotemporal Variations* (IJRR 2024) when you use them.
