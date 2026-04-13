= ./no-accum.rl
# 
# Shared comparison config for the C/C++/Python binding examples that are used
# to compare against pelorus_pico.
#
# pelorus_pico consumes raw gyro/accel samples directly and does not implement
# Pelorus' optional imu-fusion AHRS preprocessing, so we disable that layer here
# to keep the comparison at the same processing stage.
imu.ahrs_filtering.method = Disable
