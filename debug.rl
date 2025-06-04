_log_level = "info"

_rules = ./../minot/testing.rl
# _winds = ["turbine_ratpub", "pelorus"]

_start_locked = true

_bag.{
	lidar.{
		_short = l
		_topic = /ouster/points
		_type = cloud
	}
	imu.{
		_short = i
		_topic = /ouster/imu
		_type = Imu
	}
}

pf! i, l 1 tick_end
pf! l 1 tick_end
pf! i 10 tick_end

reset! ./../minot/dlg_cut
# pf! [i, l] ..200ms 0.3 # imu ekf initialization

pf! i, l 1
# pf! i 10
pf! l 1

