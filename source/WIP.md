# Draw Things i4x tool

Package the working Qwen encoder converter as a shareable macOS command-line project. Required interface: input checkpoint argument, optional output directory, default output directory equal to input directory. Preserve original files and avoid changing Draw Things registration or configuration.

The reference converter and an ongoing full Qwen3-VL 8B conversion are in the separate `../DrawThings-encoder-i4x` workdir. That job must not be interrupted. Reuse its pinned upstream ccv codec implementation and independently verified format handling.

The primary executable is a standalone Swift/C Mach-O binary targeting macOS 13+ on the host architecture (Apple Silicon tested). It has no Python or other non-system runtime dependency and no self-extraction. Build with `./script/build_and_run.sh --build-only`.

Interface: `dt-i4x INPUT.ckpt [-o OUTPUT_FOLDER]`; omitted folder defaults to the input's folder. `--inspect` checks codecs without writing, `--all-matrices` explicitly includes other models, and `--no-hadamard` disables recognized Qwen linear-operator rotation. Existing checkpoints are never overwritten. No Draw Things registration or configuration files are touched.

Supported source matrices: raw FP16/FP32/FP64/BF16, native i8x, EZM7 FP16, and palette q4p/q5p/q6p/q7p/q8p. Both trailer stores and external sidecars are supported, including SQLite tensor splits. Other selected codecs fail preflight. Unselected payloads are preserved.

The Python implementation is retained as a reference. Thirteen tests pass, including native-versus-reference packed payload equality for all source codec classes, default/explicit output paths, no-clobber, sidecars, split tensors, and preservation of unselected tensors. The finished 8B encoder conversion is in the separate original workdir and has passed all 856 tensor checks.
