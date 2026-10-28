# Draw Things quantizer (`dt-i4x`)

A macOS command-line tool for quantizing native Draw Things checkpoints. It reuses the upstream ccv CPU quantizers. Single-file safetensors input is imported through the official Draw Things CLI, which detects the architecture and remaps PyTorch tensor names.

## Use

```sh
./dt-i4x diffusion_model_f16.ckpt
./dt-i4x diffusion_model_f16.ckpt --format i3x
./dt-i4x diffusion_model.safetensors --format iq2xxs -o converted
./dt-i4x diffusion_model_f16.ckpt --inspect
```

i4x (native Q4K) is the default. Output goes beside the input unless `-o DIR` is given. Every eligible 2D matrix in the supplied checkpoint is selected, including diffusion, vision, and text components. Convolutions, biases, normalization vectors, and matrices with a singleton dimension are preserved.

The supplied file determines the scope automatically; no component-selection flags are needed. Input checkpoints and sidecars are read-only; existing outputs and validation manifests are never overwritten.

## Output formats

Use `--format NAME`; names are case insensitive, and spellings such as `Q3_K` and `i3x` are accepted.

| Format | Internal alias | Packed bits per weight before row scales/alignment |
| --- | --- | --- |
| i4x (default) | q4k | 4.5 |
| i3x | q3k | 3.5 |
| i2x | q2k | 2.625 |
| iq3s, iq3xxs | corresponding format name | 3.5 |
| iq2s, iq2xs | corresponding format name | 2.625 |
| iq2xxs | iq2xxs | 2 |
| i5x | q5k | 5.5 |
| i6x | q6k | 6.5 |

These are Draw Things native packed rowwise formats in the i8x codec family, not GGUF files. Group metadata is included in the figures above; row scales, alignment, and preserved tensors add to the final size. Lower bit counts trade weight precision for storage. Model loading and generation quality depend on the Draw Things runtime and model-specific loader.

Legacy q4p-q8p palettes are accepted as inputs. This tool does not write legacy palette formats. Original floating-point weights are preferable to requantizing compressed weights.

## Hadamard and progress

H256 is automatic only for recognized Qwen-style projection operators in text-model namespaces whose input width is divisible by 256. Embeddings, diffusion/vision operators, unknown operators, and incompatible widths stay unrotated. This deliberately keeps the established operator allowlist rather than guessing compatibility from matrix dimensions. `--no-hadamard` disables H256 globally and has no effect on matrices already using unrotated packing. Native `--inspect` reports the automatic H256/unrotated matrix counts, respecting that flag.

Progress goes to stderr and shows completed/total matrices, percentage of selected parameters, elapsed time, and the active tensor. Assembly and hashing report byte progress, and final payload verification reports tensor progress. `-q` suppresses progress and prints only the finished main checkpoint path.

## Safetensors input

Install the official `draw-things-cli` separately, or provide `--draw-things-cli /path/to/draw-things-cli`. Native checkpoint conversion has no importer dependency.

The tool runs `draw-things-cli models import` offline, with companion downloads disabled, inside `<output>.import.work/models`. Supported architectures and input layouts are those recognized by that installed importer. A single `.safetensors` model artifact is accepted; Diffusers directories and sharded index files are not assembled by this tool. Unsupported architectures fail through the official importer rather than receiving guessed tensor mappings.

The imported main checkpoint is quantized. Any companion checkpoints extracted by the importer are copied unchanged beside it. A separate `<output-stem>.model.json` preserves the inferred model specification with its main filename updated. Import does not register the result in your installed Draw Things app or edit its configuration. Register the result yourself; missing companions must already be available or supplied separately.

For safetensors, `--inspect` uses the upstream importer's dry run to report the detected model and projected files. It does not import or quantize weights. Native `--inspect` returns codec and matrix-selection JSON.

## Native input support

Raw FP16/FP32/FP64/BF16, rowwise i8x, EZM7 FP16, q4p-q8p palettes, and all packed formats listed above (including H256 variants) are supported. Trailer stores, SQLite tensor splits, and `-tensordata` sidecars are supported. Keep sidecars beside their checkpoints. Unsupported selected codecs or datatypes fail preflight. Unselected payloads are preserved.

## Verification and temporary files

Each converted tensor is decoded and checked for finite values and sampled relative weight RMSE, with the existing 0.30 error limit. Aggressive lower-bit formats can fail that limit on difficult tensors. Final tensor payload hashes, SQLite integrity, and the whole-file SHA-256 are checked. Results are recorded in a `.validation.json` manifest; these checks do not establish generation quality.

Native conversion stages files in `<output>.work`; the executable removes that directory on success. Safetensors import also uses `<output>.import.work` and removes its generated full-precision import after success. On failure, generated work directories remain for inspection and the original input stays unchanged. Remove the specific work directory before retrying that output path. Allow approximately twice the quantized output size for native conversion, plus the full-precision import and any extracted companions for safetensors.

## Build

The executable targets macOS 13+ on the host architecture (Apple Silicon tested). Building requires Xcode or Apple Command Line Tools:

```sh
./script/build_and_run.sh --build-only
./dist/dt-i4x --help
```

The Python reference implementation is retained, requires NumPy and a compiler for its first native codec build, and shares the CLI behavior. Install it with `python3 -m pip install .`. The Swift executable needs only system libraries/frameworks for native checkpoint conversion.

## Upstream and license

Quantization and palette decoding reuse [liuliu/ccv](https://github.com/liuliu/ccv), revision `810fb2992bd4b5463052a8f94338c8bffc273cc6`. Format identifiers and checkpoint layouts follow [liuliu/s4nnc](https://github.com/liuliu/s4nnc), revision `312811284158240406da8d16e30d22c636df9f29`. Architecture detection and safetensors mapping use the installed [official Draw Things CLI](https://github.com/drawthingsai/draw-things-community/blob/main/Apps/DrawThingsCLI/DrawThingsCLI.swift).

This project is BSD-3-Clause licensed and independent of the Draw Things app. Upstream BSD notices are included. Model weights are not included.
