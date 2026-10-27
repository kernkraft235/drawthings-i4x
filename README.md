# Draw Things i4x

A small, standalone macOS command-line tool that converts native Draw Things checkpoint matrices to **4-bit S (`i4x`)**. Built with Swift and the upstream ccv CPU quantizer. No Python, NumPy, compiler, app installation, model download, or network connection is required to run the executable.

The initial use case is the Qwen3-VL 8B text encoder used by Qwen Image 2.1. Its 8.18 GiB native i8x checkpoint converts to a 4.51 GiB file while preserving the vision tower and small tensors.

## Use

Unzip the macOS release and run its `dt-i4x` executable:

```sh
./dt-i4x qwen_3_vl_8b_instruct_i8x.ckpt
```

The output is `qwen_3_vl_8b_instruct_i4x.ckpt`, in the same folder as the input. To choose another folder:

```sh
./dt-i4x qwen_3_vl_8b_instruct_i8x.ckpt -o converted
```

Inspect a checkpoint before converting:

```sh
./dt-i4x qwen_3_vl_8b_instruct_i8x.ckpt --inspect
```

Default selection is 2D matrices inside `__text_model__` and numbered text-model namespaces. Norms, biases, small vectors, and the vision/diffusion models are preserved. `--all-matrices` explicitly selects every floating-point 2D matrix instead. `--no-hadamard` disables rotation. `-q` prints only the finished output path.

Input files and sidecars are read-only. Existing output checkpoints are never overwritten. The tool does not edit `custom.json`, model registration, app settings, or any other Draw Things files. Register the resulting checkpoint yourself.

## Input support

| Source representation | Supported |
| --- | --- |
| Raw FP16, FP32, FP64, BF16 | Yes |
| Native rowwise i8x | Yes |
| DT palette q4p, q5p, q6p, q7p, q8p | Yes |
| EZM7-compressed FP16 | Yes |
| Single-file trailer stores | Yes |
| SQLite plus `-tensordata` sidecar | Yes, keep the sidecar beside its checkpoint |
| SQLite tensor splits | Yes |
| Other selected codecs, including existing packed i4x/i5x/i6x | Rejected before output is created |
| Safetensors, GGUF, PyTorch checkpoints | No; import into native DT format first |

The source does **not** need to be i8x. Original floating-point weights are preferable to requantizing existing compressed weights. Unsupported codecs in tensors that are not selected for conversion are preserved unchanged.

Palette block sizes must be divisible by eight. Q5/Q7 tails must contain a multiple of eight weights, and Q6 tails a multiple of four; unusual partial tails are rejected before native decoding.

The output uses DT's own rowwise Q4K representation, internally exposed by s4nnc as `.i8x(.q4k)`. This is **not a GGUF Q4_K file**. H256 rotation is applied only to recognized Qwen linear operators with suitable dimensions; token embeddings and other operators use plain Q4K. Older model-specific DT loaders must also accept the i8x codec family to load a new i4x checkpoint. Storage conversion alone does not change the app's loading policy.

## Verification and output

Each converted tensor is decoded with the upstream decoder and checked for finite values and sampled relative weight error. The final SQLite database, tensor names, every payload hash, and whole-file SHA-256 are checked. A `.validation.json` manifest is written alongside the checkpoint.

Conversion stages files inside `<output>.work`. On success that generated directory is removed. On interruption/failure, the original input stays unchanged and the work directory remains for inspection; remove that specific generated directory before retrying the same output path. Allow disk space for approximately twice the final checkpoint size during conversion.

The Qwen3-VL 8B example converted all 253 text matrices (7,568,097,280 weights); all 856 tensors were verified, with the other 603 tensors preserved. Runtime image quality and device memory behavior are evaluated by loading the file in Draw Things.

## Build

The included release binary targets **Apple Silicon, macOS 13+**. Building requires Xcode or Apple Command Line Tools:

```sh
./script/build_and_run.sh --build-only
./dist/dt-i4x --help
```

The script uses the host architecture. The build has been tested on Apple Silicon. The native executable links only system libraries and frameworks; it does not unpack a runtime into a temporary directory.

## Tests and reference implementation

The Python implementation is retained as a reference and can be installed separately with `python3 -m pip install .`. It requires NumPy and a compiler for its first native codec build. The shipped Swift executable has neither dependency.

After building the executable, run the cross-implementation tests:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover output-directory defaults, no-clobber behavior, unsupported inputs, primitive source types, i8x, EZM7, all five palette formats, preserved vision tensors, and exact native-versus-reference packed payloads. Independent NumPy decoding of official i4x/H256 data was also used to validate the codec before the full encoder conversion.

## Upstream and license

Quantization implementation and tables: [liuliu/ccv](https://github.com/liuliu/ccv), revision `810fb2992bd4b5463052a8f94338c8bffc273cc6`. The palette decoder is extracted from the same revision. The compatibility headers adapt allocation and IEEE floating-point conversion for a standalone CPU build. Upstream BSD license is included in `src/dt_i4x/native/UPSTREAM-LICENSE.txt`.

Checkpoint identifiers, external-data layouts, and the trailer-store offset follow [liuliu/s4nnc](https://github.com/liuliu/s4nnc), revision `312811284158240406da8d16e30d22c636df9f29`.

This project is BSD-3-Clause licensed and is independent of the Draw Things app. Model weights are not included.
