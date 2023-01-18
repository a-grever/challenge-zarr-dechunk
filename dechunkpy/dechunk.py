from __future__ import annotations

import argparse
import math
import os
import pathlib
import shutil
from typing import Any, Iterator, Literal

import numcodecs
import numpy as np
import pydantic

parser = argparse.ArgumentParser()

parser.add_argument("filename", type=pathlib.Path)
parser.add_argument("-r", "--replace", action="store_true")


class CompressorConfig(pydantic.BaseModel):
    blocksize: int
    clevel: int
    cname: str
    id: str
    shuffle: int


class ArrayMetaData(pydantic.BaseModel):
    chunks: tuple[int]
    compressor: CompressorConfig
    dtype: str
    fill_value: Any = 0
    filters: Any = None
    order: Literal["C", "F"]
    shape: tuple[int]
    zarr_format: Literal[2]

    @classmethod
    def from_path(cls, metadata_path: str | pathlib.Path) -> ArrayMetaData:
        metadata_path = pathlib.Path(metadata_path)
        with metadata_path.open(mode="r") as f:
            return cls.parse_raw(f.read())


class Dechunker:
    def __init__(self, *, array_path: pathlib.Path, write_chunks: int = 50):
        self.array_path = array_path
        assert (
            array_path / ".zarray"
        ).exists(), "missing .zarray metadata file in provided array path"
        ArrayMetaData.update_forward_refs()
        self.metadata = ArrayMetaData.from_path(array_path / ".zarray")
        self.compressor = numcodecs.get_codec(self.metadata.compressor.dict())
        self.write_chunks = write_chunks

    @property
    def new_path_folder(self) -> pathlib.Path:
        folder = self.array_path.parent / (self.array_path.name + ".new")
        if folder.exists():
            return folder
        folder.mkdir()
        return folder

    @property
    def new_path_metadata(self) -> pathlib.Path:
        file = self.new_path_folder / ".zarray"
        file.touch()
        return file

    @property
    def new_path_single_chunk(self) -> pathlib.Path:
        file = self.new_path_folder / "0"
        file.touch()
        return file

    @property
    def new_path_mmmap(self) -> pathlib.Path:
        file = self.new_path_folder / "0.mmap"
        file.touch()
        return file

    def write_dechunked_metadata(self):
        with self.new_path_metadata.open(mode="w") as f:
            f.write(
                ArrayMetaData(
                    **{
                        **self.metadata.dict(),
                        "chunks": [self.metadata.shape[0]]
                    }
                ).json(indent=4)
            )

    def array_chunks(self) -> Iterator[Any]:
        n_items = self.metadata.shape[0]
        chunk_size = self.metadata.chunks[0]
        n_items_left = n_items
        n_chunks = math.ceil(n_items / chunk_size)
        for i in range(n_chunks):
            with open(self.array_path / f"{i}", mode="rb") as f:
                mem = memoryview(self.compressor.decode(f.read()))
                chunk = np.array(mem, copy=False)
                chunk_serialized = chunk.view(self.metadata.dtype)
                yield chunk_serialized[:n_items_left]
                n_items_left -= len(chunk_serialized)

    def cleanup(self):
        for new_file in self.new_path_folder.iterdir():
            new_file.unlink()
        self.new_path_folder.rmdir()

    def replace(self):
        temp_folder = self.array_path.parent / (self.array_path.name + ".temp")
        os.replace(self.array_path, temp_folder)
        os.replace(self.new_path_folder, self.array_path)
        shutil.rmtree(temp_folder)

    def dechunk(self, replace=False):
        try:
            self.cleanup()

            fp = np.memmap(
                self.new_path_mmmap,
                dtype=self.metadata.dtype,
                mode="w+",
                shape=self.metadata.shape,
            )
            offset = 0
            for chunk in self.array_chunks():
                size = len(chunk)
                fp[offset: offset + size] = chunk
                offset += size

            with self.new_path_single_chunk.open(mode="bw") as f:
                f.write(self.compressor.encode(fp))

            self.write_dechunked_metadata()
            self.new_path_mmmap.unlink()

            if replace:
                self.replace()
        except Exception:
            self.cleanup()
            raise


def main():
    args = parser.parse_args()
    dechunker = Dechunker(array_path=args.filename)
    dechunker.dechunk(replace=args.replace)


if __name__ == "__main__":
    main()
