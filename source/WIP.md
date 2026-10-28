# Draw Things quantizer

The Swift/C executable supports native packed Q2K/Q3K/Q4K/Q5K/Q6K and IQ2/IQ3 outputs through `--format`; q4k/i4x remains the default. Matrix selection automatically covers eligible matrices in the supplied file, without component-selection flags. H256 remains automatic only for the established text-projection allowlist with compatible dimensions; `--no-hadamard` disables it globally. Progress reports parameter percentages, matrix counts, elapsed time, and assembly/verification/hash stages.

Single-file safetensors input delegates architecture detection and tensor mapping to `draw-things-cli models import`, offline and isolated from the app models directory. The quantized main model, extracted companions, and a separate model specification are emitted together. The Python reference exposes the same options.

Build with `./script/build_and_run.sh --build-only`; the development binary is `dist/dt-i4x`. The repository-root executable is the user-facing copy. Original inputs and Draw Things app configuration are not modified. Runtime generation quality remains a user check in Draw Things.
