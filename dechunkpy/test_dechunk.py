import pathlib

import numpy as np
import zarr
from dechunk import Dechunker

TEST_FOLDER = pathlib.Path(__file__).parent


def test_dechunk_integer_array_chunk_size_1(tmp_path: pathlib.Path):

    a = zarr.zeros((10), chunks=(1), dtype="uint16")
    a[:] = np.arange(10)
    path = tmp_path / "test.zarr"
    zarr.save(path, a)

    before = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 11

    Dechunker(array_path=path).dechunk(replace=True)

    after = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 2
    assert all(before == after)


def test_dechunk_integer_array_chunk_size_3(tmp_path: pathlib.Path):

    a = zarr.zeros((10), chunks=(3), dtype="uint16")
    a[:] = np.arange(10)
    path = tmp_path / "test.zarr"
    zarr.save(path, a)

    before = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 5

    Dechunker(array_path=path).dechunk(replace=True)

    after = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 2
    assert all(before == after)


def test_dechunk_datetime_ns_array_chunk_size_1(tmp_path: pathlib.Path):
    a = zarr.zeros((10), chunks=(3), dtype="<M8[D]")
    start_ts = np.datetime64("2022-09-22T16:52:35.000000000")
    a[:] = np.array(
        [
            start_ts + np.timedelta64(i, "D")
            for i in np.arange(10)
        ]
    )
    path = tmp_path / "test.zarr"
    zarr.save(path, a)

    before = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 5

    Dechunker(array_path=path).dechunk(replace=True)

    after = zarr.open(path)[:]
    assert len((list(path.iterdir()))) == 2
    assert all(before == after)
