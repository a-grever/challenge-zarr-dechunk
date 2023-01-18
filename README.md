# challenge-zarr-dechunk
This repository contains the same script in two languages, [Python](./dechunkpy) and [Rust](./dechunkrs/). The script takes a chunked a 1D [zarr array](https://zarr.readthedocs.io/en/stable/index.html) and reformats it as a single chunk array. The steps involved are:
* iterating over the compressed chunks
* decompress according to the array metadata
* combine bytes of the decompressed chunks, cutting of the fill value at the end 
* compress the single chunk and store as a chunk file in a tmp folder
* save metadata with updated chunk size (number of elements) in a tmp folder
* replace the existing array folder with the tmp folder

## Assumptions
* only 1D arrays, no update of groups
* zarr format version: 2

## Python implementation
The Python implementation uses the `numcodecs` package for (de)compression of the data chunk(s).

* setup your virtual environment, e.g. via
    ```
    python -m venv venv
    source ./venv/bin/activate   
    pip install --upgrade pip
    pip install -r dechunkpy/requirements.txt
    ```
* run the script
    ```
    python dechunkpy/dechunk.py [-r] <path/to/array/folder.zarr> 
    ```
    To replace the existing file run with the `-r/--replace`

* run tests
    ```
    pytest dechunkpy
    ```

## Rust implementation
The Rust implementation only supports compression via `blosc`.
Run the script via
```
cargo run <path/to/array/folder.zarr> 
```
