<div align="center">

# AgentRobustBench

**English** | [简体中文](README_zh-CN.md)

**A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents**

NeurIPS 2026 · Evaluations & Datasets Track

[![Paper on arXiv](https://img.shields.io/badge/arXiv-2605.11928-b31b1b)](https://arxiv.org/abs/2605.11928)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench)
[![Live leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-7C3AED)](https://huggingface.co/spaces/willchow66/AgentRobustBench)

**199 seed tasks · 22 perturbation types · 3,721 evaluation cases · 21 models evaluated**

[Run the checkpoint](#run-six-examples) · [Benchmark](#benchmark) · [Results](#results) · [Full evaluation](docs/REPRODUCIBILITY.md) · [Checkpoints](#model-checkpoints) · [Training](#reproducing-toolrl-dr-optional) · [Citation](#citation)

</div>

AgentRobustBench measures how tool-use language agents handle failures that arise in deployment, from typos and ambiguous tool names to API timeouts. It pairs a robustness benchmark with **ToolRL-DR**, a reinforcement learning recipe that trains agents on varied tool-use trajectories.

## Run six examples

This runs the released checkpoint on **three clean tasks and their three typo variants**, then scores the generated tool calls. Choose where the model will run:

| Your computer | How to run |
| --- | --- |
| CPU only, including an Intel Mac | Direct inference with `--device cpu`; no GPU or model server needed. |
| Apple Silicon Mac | Use its built-in GPU through **MPS** with `--device auto`, when macOS and the installed PyTorch support it. No NVIDIA GPU or CUDA is needed. |
| Windows/Linux with an NVIDIA GPU | Use **CUDA** with `--device auto`. Install a compatible NVIDIA driver and CUDA-enabled PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/). The [vLLM guide](scripts/README.md) is another option on supported Linux machines. |
| Windows with an Intel or AMD GPU | The current runner has no dedicated adapter for these GPU backends. Use `--device cpu` for this workflow. |
| Laptop connected to an existing model server | Install only the [evaluation client](scripts/README.md#if-you-already-have-a-model-server). The server does the model inference. |

A supported GPU can make **model generation faster than CPU inference**. The speedup depends on the GPU, available memory, and input/output lengths; it has not been measured for these checkpoints yet. Rule-based scoring runs on CPU, and choosing a GPU does not speed up the initial download.

The HF weights are about **13.6 GB per checkpoint**. CPU inference uses float32 and needs additional RAM beyond the weights; **32 GB RAM is a practical starting recommendation, not a measured minimum**. Accelerator inference uses half precision. Download speed and inference time depend on your computer. See [validation status](docs/REPRODUCIBILITY.md) for what has been tested.

### 1. Install

Install [Python 3.11 or 3.12](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads). Open Terminal on macOS/Linux or PowerShell on Windows. Download the project, then create a separate environment:

```bash
git clone --depth 1 https://github.com/WillChow66/AgentRobustBench.git
cd AgentRobustBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-local.txt
```

Already have the repository? Open its directory and skip the clone. On Windows, replace the last three commands with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-local.txt
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `python` in the commands below. For a CPU-only Linux/Windows installation, install the CPU build of PyTorch before the requirements: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`. macOS uses the normal installation above.

### 2. Check your environment

```bash
python scripts/evaluate_checkpoint.py --device auto --check
```

This prints your platform, selected device, available system RAM, and package versions **without downloading weights**. Read the `Device:` line:

| Output | Device used for model generation |
| --- | --- |
| `Device: mps` | The Mac GPU through Apple MPS. |
| `Device: cuda` | The NVIDIA GPU through CUDA. |
| `Device: cpu` | The CPU; no supported accelerator was detected. |

`auto` tries CUDA first, then MPS, then CPU. If you expected CUDA but see `cpu`, check that your NVIDIA driver and CUDA-enabled PyTorch are installed. On Mac, check MPS availability with your macOS/PyTorch versions. To explicitly select a device, replace `auto` with `cpu`, `mps`, or `cuda`; an unavailable explicit device produces an error.

### 3. Run Full or Mixed

```bash
python scripts/evaluate_checkpoint.py --checkpoint full --device auto
```

For Mixed, run:

```bash
python scripts/evaluate_checkpoint.py --checkpoint mixed --device auto
```

Both commands automatically load the matching HF tokenizer and checkpoint, generate predictions using the benchmark's prompts and parsers, and run the existing scorer. The data comes with the repository. After the first download, HF normally reuses its local cache. These commands use the device selected by the check above. To force CPU inference, replace `--device auto` with `--device cpu`.

### 4. Read the results

The scorer prints **`n=6`** and the accuracy measured from your model outputs. The last line prints a new directory under `results/` containing:

| File | Contents |
| --- | --- |
| `clean.predictions.jsonl`, `realistic_typos.predictions.jsonl` | Generated text and parsed tool calls. |
| `scores.json` | Aggregate accuracy. |
| `per_sample_scores.jsonl` | Score for each example. |
| `_run_summary.json` | Processed counts and errors. |
| `run_metadata.json` | Checkpoint revision, data hashes, device, precision, package versions, seed, and measured load/evaluation times. |

The six-example run checks the evaluation workflow. To run every case and compare newly measured scores with the released scores:

```bash
python scripts/evaluate_checkpoint.py --checkpoint full --suite full --device auto
python scripts/evaluate_checkpoint.py --checkpoint mixed --suite full --device auto
```

A complete run scores **3,721 cases** and also writes `comparison.csv` with measured scores, released scores, and differences in percentage points. See [reproduction details](docs/REPRODUCIBILITY.md) and [the vLLM guide](scripts/README.md) for other serving setups.

## Benchmark

![Diagram of the four perturbation categories between a tool-use agent and its environment](docs/assets/benchmark-overview.webp)

| Category | What changes | Example |
| --- | --- | --- |
| Observation | The input the agent sees | A typo in the user query |
| Action | The tools available to the agent | A distractor tool with the same name |
| Reward-relevant metadata | Information that steers tool choice | A misleading tool name or response-time annotation |
| Transition | The response to a tool call | A transient timeout on the first attempt |

The [Hugging Face dataset](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench) contains **2,718 static records**. The default single-turn runner evaluates **2,527** of them, skipping **191 multi-turn records**, and adds **1,194 runtime cases** (199 clean tasks × 6 error types), for **3,721 evaluated cases** in total. Submit predictions or browse per-category scores on the [live leaderboard](https://huggingface.co/spaces/willchow66/AgentRobustBench).

## Results

![Robustness retention across model sizes, with ToolRL-DR-Full and ToolRL-DR-Mixed highlighted](docs/assets/robustness-retention.webp)

The plot shows how much clean accuracy models retain across Observation, Action, and Reward perturbations. At 3B parameters, ToolRL-DR-Full improves retention over the size-matched ToolRL baseline. In a separate runtime evaluation, it closes about 27% of the Transition gap without training on Transition errors. See the [paper](https://arxiv.org/abs/2605.11928) for the metric and experimental settings.

## Layout

| Directory | Contents |
|-----------|----------|
| `hf_data/datasets/api_eval/` | 17 JSONL files: 16 static perturbation cells + the clean baseline (3,721 predictions per model in total). The 6 Transition variants are produced at runtime by `scripts/run_eval.py`, not stored as separate files. |
| `scripts/` | Direct checkpoint runner (`evaluate_checkpoint.py`), endpoint runner (`run_eval.py`), reparser (`reparse_predictions.py`), model manifest, and SLURM templates. See `scripts/README.md`. |
| `leaderboard/` | HuggingFace Space implementation: Gradio app, deterministic scorer, leaderboard CSV. See `leaderboard/README.md`. |
| `training/` | Reproduction recipe for ToolRL-DR-Full and ToolRL-DR-Mixed: the parquets actually fed to the GRPO trainer (`train.parquet`, `mixed_train.parquet`, `test.parquet`), the reference jsonl→parquet converter, our drop-in reward-score patch (`rlla_perturbed.py`), the GRPO launcher, and a hyperparameter manifest. See `training/README.md`. |

## Model checkpoints

| Checkpoint | Training data | Model card |
| --- | --- | --- |
| ToolRL-DR-Full | Perturbed trajectories | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Full-Qwen2.5-3B) |
| ToolRL-DR-Mixed | Clean and perturbed trajectories | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Mixed-Qwen2.5-3B) |

The checkpoints are based on Qwen2.5-3B-Instruct and use the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE). See [training/](training/) for the training data and reproduction instructions.

## Full evaluation with a model server

Start with [six examples](scripts/README.md#run-six-examples), then follow [full benchmark evaluation](scripts/README.md#full-benchmark-evaluation) to run all static splits and the six runtime perturbations. The guide includes the install commands, model-server check, inference commands, scoring command, and output locations.

For cluster users, the SLURM pipeline is in [`scripts/slurm/`](scripts/slurm/). It needs your cluster's account, paths, and scheduler settings.

## Reproducing ToolRL-DR (optional)

The parquets fed to the GRPO trainer are already in `training/data/`,
so you can train without running the data converter. After the one-time setup
described in `training/README.md` (clone public ToolRL, drop in the
`rlla_perturbed.py` reward patch, set `PYTHONPATH`):

```bash
# DR-Full
TRAIN_FILE=training/data/train.parquet \
VAL_FILE=training/data/test.parquet \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-full \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh

# DR-Mixed: change TRAIN_FILE to training/data/mixed_train.parquet
```

## Citation

The current arXiv record lists the paper under its original title.

```bibtex
@misc{zhou2026whensimulationlies,
  title         = {When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents},
  author        = {Zhou, Xiaolin and Yuan, Aojie and Luo, Zheng and Ling, Zipeng and
                   Pan, Xixiao and Gao, Yicheng and Zhang, Haiyue and Li, Jiate and
                   Jiang, Shuli and Wang, Prince Zizhuang and Zhu, Zixuan and
                   Liu, Jinbo and Rossi, Ryan A. and Wei, Hua and Hu, Xiyang},
  year          = {2026},
  eprint        = {2605.11928},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## License

Code: [Apache-2.0](LICENSE). Data: see per-source license documented in the paper.
