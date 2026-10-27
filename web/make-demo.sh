#!/usr/bin/env bash
# Cut the demo bags from the start of each recording: the lidar and IMU topics
# only, without point fields Pelorus does not read, clouds compressed with
# cloudini at 1 mm precision. The bags come from marina, so the datasets must
# be in its cache or a registry.
#
#   oxford_spires/keble_college_02    Oxford Spires, handheld (CC BY-NC-SA 4.0)
#   grand_tour/<mission>:hesai:stim320  GrandTour, ANYmal (MIT)
#   helipr/<sequence>:ouster          HeLiPR, car. Its license is not stated, so
#                                     confirm it with the authors before hosting.
#
# The HeLiPR Ouster OS2-128 clouds are about 120k points, so those demos keep
# every third point and 20 seconds to stay below 80 MB.
#
# The CI deploys the bags from the marina dataset pelorus/web_demo. After
# changing them, push them there without recompressing the clouds:
#
#   marina push --pointcloud-mode off pelorus/web_demo web/demo
#
# Usage: web/make-demo.sh
set -euo pipefail

web="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$web")"

cargo build --manifest-path "$root/Cargo.toml" -p pelorus_wasm --release --example demo_bag
# cut <dataset> <bag file> <output> <cloud topic> <imu topic> [demo_bag options]
cut() {
  local dataset="$1" file="$2" output="$3" cloud="$4" imu="$5"
  shift 5
  "$root/target/release/examples/demo_bag" \
    "$(marina resolve "$dataset")/$file" "$web/demo/$output" \
    --cloud "$cloud" --imu "$imu" --precision 0.001 "$@"
}

mkdir -p "$web/demo"
cut oxford_spires/keble_college_02 keble_college_02_0.mcap keble_college_02_30s.mcap \
  /hesai/pandar /alphasense_driver_ros/imu --duration 30 --drop-fields ring
for mission in sbb1 eth1 snow1; do
  cut "grand_tour/$mission:hesai:stim320" ros2_hesai_stim320_0.mcap "grand_tour_${mission}_30s.mcap" \
    /boxi/hesai/points /boxi/stim320/imu --duration 30 --drop-fields ring
done
cut helipr/town02:ouster Town02_ouster_0.mcap helipr_town02_20s.mcap \
  /ouster/points /imu/data_raw --duration 20 --drop-fields ring,reflectivity,ambient --keep-every 3
cut helipr/bridge02:ouster Bridge02_ouster_0.mcap helipr_bridge02_20s.mcap \
  /ouster/points /imu/data_raw --duration 20 --drop-fields ring,reflectivity,ambient --keep-every 3
