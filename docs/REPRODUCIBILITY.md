# Checkpoint evaluation and reproduction

[English quick start](../README.md#run-six-examples) · [中文入门](../README_zh-CN.md#运行六个模型评测样本)

## Run the released models

`scripts/evaluate_checkpoint.py` loads the model itself. It uses the existing
benchmark prompts, tool-call parsers, transition injection, and deterministic
scorer. It runs generation sequentially to limit memory use. vLLM is optional.

| Argument | Model loaded from HF |
| --- | --- |
| `--checkpoint full` | `willchow66/ToolRL-DR-Full-Qwen2.5-3B` |
| `--checkpoint mixed` | `willchow66/ToolRL-DR-Mixed-Qwen2.5-3B` |

The exact HF revisions are stored in [`configs/checkpoints.json`](../configs/checkpoints.json).
Both repositories were checked as public and ungated on 2026-11-09. Each has
three safetensors shards totaling 13,588,414,464 bytes. HF downloads them once
and normally reuses its cache. For a local copy, use `--model-path /path/to/model`.

## Device and precision

| Option | Generation device | Precision |
| --- | --- | --- |
| `--device cpu` | CPU | float32 |
| `--device mps` | Apple Silicon GPU, when available | float16 |
| `--device cuda` | NVIDIA GPU, when available | bfloat16 if supported, otherwise float16 |
| `--device auto` | CUDA first, then MPS, then CPU | As above |

CPU inference needs RAM for approximately 13.6 GB of float32 weights plus
activations, attention caches, and other processes. The README's 32 GB RAM
recommendation is a planning estimate, not a measured minimum. Accelerator
memory for the weights is approximately 6.8 GB in half precision, with extra
memory needed for inference. Peak RAM/VRAM and the released models' runtime
have not yet been measured here. There is no five-minute runtime promise.

Generation defaults are temperature 0, maximum 1,024 new tokens, context limit
8,192, one worker, and seed 42. This seed is a documented default for the new
local runner, not a claim about the original training seed. The original vLLM
model-specific settings remain in [`scripts/model_manifest.py`](../scripts/model_manifest.py).
Different precision and inference backends can change generated predictions.

## From a small run to full results

```bash
# Three clean tasks and the same three typo tasks; generates and scores six outputs.
python scripts/evaluate_checkpoint.py --checkpoint full --device cpu

# Full evaluation, including all six runtime error types.
python scripts/evaluate_checkpoint.py --checkpoint full --suite full --device auto
python scripts/evaluate_checkpoint.py --checkpoint mixed --suite full --device auto
```

The full suite uses 2,527 static single-turn records plus 199 × 6 runtime cases,
for 3,721 scored cases. It skips the 191 multi-turn records in the static files.
Every run gets a new directory unless `--output-dir` is supplied. A nonempty
output directory is rejected to avoid mixing separate experiments.

## Metric mapping

Full evaluation writes `comparison.csv`. Its **measured** column comes from
new model predictions; its **released** column is a reference read from
[`leaderboard/leaderboard.csv`](../leaderboard/leaderboard.csv).

| Column / metric | Calculation from newly scored cases |
| --- | --- |
| `Clean` | Correct clean cases divided by 199. |
| `Pert. Acc.` | Correct perturbed cases divided by all perturbed cases, excluding clean. |
| `Obs Acc.` | Accuracy on observation perturbations. |
| `Action Acc.` | Accuracy on action perturbations. |
| `Reward Acc.` | Accuracy on reward-relevant perturbations. |
| `Transition Acc.` | Accuracy on the six runtime error types. |
| Clean-to-Transition gap | `Clean` minus `Transition Acc.`. |

`comparison.csv` reports measured minus released in percentage points. It does
not replace newly measured scores with the reference values. The two checkpoint
runs cover their benchmark scores; reproducing comparisons with the ToolRL
baseline or all other models also requires running those models using the
[endpoint guide](../scripts/README.md). Figure-generation scripts are not yet
included in this release.

## Recorded run information

`run_metadata.json` records the checkpoint and code revisions, dataset file
hashes, Python and package versions, device, dtype, generation settings, load
time, evaluation time, total time, and completion status. Predictions and
per-sample scores remain alongside it. Generation errors stop the run before
scoring, and partial outputs can be inspected.

## Validation status

The CPU development environment uses Python 3.12, PyTorch 2.15.0+cpu,
Transformers 4.57.6, Accelerate 1.15.0, OpenAI SDK 3.48.0, and tqdm 4.70.1.
The matching constraints are in
[`scripts/constraints-local-cpu.txt`](../scripts/constraints-local-cpu.txt).
These are CPU workflow test versions, not the original paper's training stack.

| Check | Scope |
| --- | --- |
| Real Transformers CPU integration test | Creates a tiny Qwen2 model locally, loads its safetensors and tokenizer, generates six predictions, and runs the benchmark scorer. No downloaded model weights or stored leaderboard scores are used for this test. |
| Failure handling | Tests unavailable devices, generation errors, and rejection of nonempty output directories. |
| Released Full/Mixed checkpoints | Public model metadata, architecture, revisions, and weight files checked; full-weight model inference is not yet validated here. |
| Apple MPS / CUDA | Device selection is implemented; execution on those devices has not yet been validated here. |
| Training | Data and launch recipe supplied; retraining has not been repeated for this update. |

Run the checks:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same CPU tests on pushes and pull requests. It does not
download the 3B checkpoints or run the full benchmark.
