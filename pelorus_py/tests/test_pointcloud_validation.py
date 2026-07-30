import numpy as np
import pytest

import pelorus


def pointcloud2_dict(*, width=1, data=b"\0\0\0\0"):
    return {
        "header": {
            "stamp": {"sec": 1, "nanosec": 0},
            "frame_id": "lidar",
        },
        "height": 1,
        "width": width,
        "fields": [
            {
                "name": "x",
                "offset": 0,
                "datatype": 7,
                "count": 1,
            }
        ],
        "is_bigendian": False,
        "point_step": 4,
        "row_step": width * 4,
        "data": data,
        "is_dense": True,
    }


def test_handle_has_public_module_name():
    assert pelorus.Handle.__module__ == "pelorus"


def test_empty_numpy_cloud_is_rejected_before_dispatch():
    lio = pelorus.Handle(device_specific_pointcloud2=False)

    with pytest.raises(ValueError, match="cannot be empty"):
        lio.add_pointcloud(
            np.empty((0, 3), dtype=np.float32),
            timestamp_sec=1,
            timestamp_nanosec=0,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_common_numpy_dtypes_are_accepted(dtype):
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    points = np.zeros((512, 3), dtype=dtype)

    scan_id = lio.add_pointcloud(
        points,
        timestamp_sec=1,
        timestamp_nanosec=0,
    )

    assert scan_id > 0


def test_tuple_point_sequences_are_accepted():
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    points = tuple((float(i), 0.0, 0.0) for i in range(512))

    scan_id = lio.add_pointcloud(
        points,
        timestamp_sec=1,
        timestamp_nanosec=0,
    )

    assert scan_id > 0


@pytest.mark.parametrize("invalid_t", [-1.0, float("nan"), 5_000_000_000.0])
def test_invalid_point_timestamps_are_rejected(invalid_t):
    lio = pelorus.Handle(device_specific_pointcloud2=False)

    with pytest.raises(ValueError, match="finite nanosecond offset"):
        lio.add_pointcloud(
            [[0.0, 0.0, 0.0, invalid_t]],
            timestamp_sec=1,
            timestamp_nanosec=0,
        )


def test_fractional_nanosecond_offsets_are_rounded():
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    points = [[float(i), 0.0, 0.0, i + 0.5] for i in range(512)]

    scan_id = lio.add_pointcloud(
        points,
        timestamp_sec=1,
        timestamp_nanosec=0,
    )

    assert scan_id > 0


def test_empty_pointcloud2_dict_is_rejected_before_dispatch():
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    cloud = pointcloud2_dict(width=0, data=b"")

    with pytest.raises(ValueError, match="greater than zero"):
        lio.add_pointcloud(cloud)


def test_pointcloud2_dict_with_no_fields_is_rejected_before_dispatch():
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    cloud = pointcloud2_dict()
    cloud["fields"] = []

    with pytest.raises(ValueError, match="fields cannot be empty"):
        lio.add_pointcloud(cloud)


def test_undersized_pointcloud2_buffer_is_rejected_before_dispatch():
    lio = pelorus.Handle(device_specific_pointcloud2=False)
    cloud = pointcloud2_dict(width=2, data=b"\0\0\0\0")

    with pytest.raises(ValueError, match="at least 8 are required"):
        lio.add_pointcloud(cloud)
