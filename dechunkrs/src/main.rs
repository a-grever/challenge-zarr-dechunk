use blosc::{decompress_bytes, Clevel, Compressor, Context, ShuffleMode};
use math::round;
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use serde_repr::{Deserialize_repr, Serialize_repr};
use std::env;
use std::fs;
use std::fs::{remove_dir_all, rename, DirBuilder, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Deserialize, Clone, Serialize)]
enum CompressionId {
    #[serde(rename = "blosc")]
    Blosc,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
struct CompressorConfig {
    blocksize: u8,
    clevel: u8,
    cname: String,
    id: CompressionId,
    shuffle: u8,
}
impl CompressorConfig {
    fn compressor_from_config(&self) -> Context {
        let _clevel = match self.clevel {
            1 => Clevel::L1,
            2 => Clevel::L2,
            3 => Clevel::L3,
            4 => Clevel::L4,
            5 => Clevel::L5,
            6 => Clevel::L6,
            7 => Clevel::L7,
            8 => Clevel::L8,
            9 => Clevel::L9,
            _ => panic!("missing blosx clevel info in metadata"),
        };
        let _shuffle_mode = match self.shuffle {
            0 => ShuffleMode::None,
            1 => ShuffleMode::Byte,
            2 => ShuffleMode::Bit,
            _ => panic!("missing shuffle_mode info in metadata"),
        };
        let _compressor_name = match self.cname.as_str() {
            "blosclz" => Compressor::BloscLZ,
            "lz4" => Compressor::LZ4,
            "lz4hc" => Compressor::LZ4HC,
            "snappy" => Compressor::Snappy,
            "zlib" => Compressor::Zlib,
            _ => panic!("missing compressor cname info in metadata"),
        };
        let ctx = Context::new()
            .compressor(_compressor_name)
            .unwrap()
            .clevel(_clevel)
            .shuffle(_shuffle_mode);
        ctx
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
enum DimensionOrder {
    C,
    F,
}
#[derive(Serialize_repr, Deserialize_repr, PartialEq, Debug, Clone)]
#[repr(u8)]
enum ZarrFormat {
    Two = 2,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
struct ArrayMetaData {
    chunks: [u32; 1], // only 1D arrays
    compressor: CompressorConfig,
    dtype: String,
    fill_value: Option<u8>, // not true, but not used
    filters: Option<u8>,    // not true, but not used
    order: DimensionOrder,
    shape: [u32; 1], // only 1D arrays
    zarr_format: ZarrFormat,
}

#[derive(Debug, Deserialize)]
struct ArrayChunk {
    current_chunk: u32,
    n_chunks: u32,
    array_path: String,
}

impl Iterator for ArrayChunk {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk == self.n_chunks {
            return None;
        }
        let chunk_path = self.array_path.clone() + "/" + &self.current_chunk.to_string();
        self.current_chunk += 1;

        let compressed_chunk: Vec<u8> = fs::read(chunk_path).expect("Unable to read file");
        let decompressed_chunk: Vec<u8> =
            unsafe { decompress_bytes::<u8>(&compressed_chunk[..]).unwrap() };
        Some(decompressed_chunk)
    }
}

struct ArrayDechunker {
    metadata: ArrayMetaData,
    array_path: String,
    compressor: Context,
}

impl ArrayDechunker {
    fn from_path(array_path: &String) -> ArrayDechunker {
        let metadata_path: String = array_path.clone() + "/.zarray";
        let metadata_buffer = fs::read_to_string(metadata_path).expect("Unable to read file");
        let metadata: ArrayMetaData =
            serde_json::from_str(&metadata_buffer).expect("JSON does not have correct format.");

        let compressor = metadata.compressor.compressor_from_config();

        ArrayDechunker {
            metadata,
            array_path: array_path.clone(),
            compressor,
        }
    }

    fn n_bytes_from_dtype(&self) -> usize {
        if self.metadata.dtype.len() < 3 {
            panic!("invalid dtype str, too short")
        }
        if self.metadata.dtype.contains("M") | self.metadata.dtype.contains("m") {
            return 8;
        }
        let n_bytes: usize = self.metadata.dtype[2..]
            .parse::<usize>()
            .expect("unexpected format of dtype, expected int in position 3");
        n_bytes
    }

    fn dechunked_metadata(&self) -> ArrayMetaData {
        let mut dechunked_metadata = self.metadata.clone();
        dechunked_metadata.chunks = self.metadata.shape;
        dechunked_metadata
    }

    fn array_chunk_iter(&self) -> ArrayChunk {
        let n_chunks: u32 = round::ceil(
            self.metadata.shape[0] as f64 / self.metadata.chunks[0] as f64,
            0,
        ) as u32;
        ArrayChunk {
            current_chunk: 0,
            n_chunks,
            array_path: self.array_path.clone(),
        }
    }

    fn dechunk(&self) {
        let mut out_buffer: Vec<u8> = Vec::new();
        let size = self.metadata.shape[0] as usize * self.n_bytes_from_dtype();
        for chunk in self.array_chunk_iter() {
            out_buffer.extend(chunk);
        }
        out_buffer.shrink_to(size);
        let out_buffer_capped = &out_buffer[..size];

        let compressed_single_chunk: Vec<u8> = self.compressor.compress(&out_buffer_capped).into();
        self.save_files(compressed_single_chunk)
    }

    fn save_files(&self, compressed_single_chunk: Vec<u8>) {
        let path_tmp_dir = Path::new("./tmp_array.zarr");
        let path_new_array_dir = Path::new("./new_array_dir.zarr");
        let path_new_array_metadata = path_new_array_dir.join(".zarray");
        let path_new_array_chunk = path_new_array_dir.join("0");

        DirBuilder::new()
            .recursive(true)
            .create(path_new_array_dir)
            .unwrap();
        let mut single_chunk_file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path_new_array_chunk)
            .unwrap();
        single_chunk_file
            .write_all(&compressed_single_chunk)
            .unwrap();

        let mut single_chunk_metadata = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path_new_array_metadata)
            .unwrap();

        single_chunk_metadata
            .write_all(
                &to_string_pretty(&self.dechunked_metadata())
                    .unwrap()
                    .into_bytes(),
            )
            .unwrap();

        let path_original_array = Path::new(&self.array_path);
        rename(path_original_array, path_tmp_dir).expect("failed to rename original array folder");
        rename(path_new_array_dir, path_original_array)
            .expect("failed to rename new array folder to original path");
        remove_dir_all(path_tmp_dir).expect("failed to delete tmp array folder");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let array_path = &args[1];

    let array_dechunker = ArrayDechunker::from_path(&array_path);
    array_dechunker.dechunk();
}
