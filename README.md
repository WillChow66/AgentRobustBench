<div align="center">

# AgentRobustBench

**English** | [简体中文](README_zh-CN.md)

**A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents**

NeurIPS 2026 · Evaluations & Datasets Track

[![Paper on arXiv](https://img.shields.io/badge/arXiv-2605.11928-b31b1b)](https://arxiv.org/abs/2605.11928)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench)
[![Live leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-7C3AED)](https://huggingface.co/spaces/willchow66/AgentRobustBench)

**199 seed tasks · 22 perturbation types · 3,721 evaluation cases · 21 models evaluated**

[Run the checkpoint](#run-six-examples) · [Benchmark](#benchmark) · [Results](#results) · [Full evaluation](#quick-start) · [Checkpoints](#model-checkpoints) · [Training](#reproducing-toolrl-dr-optional) · [Citation](#citation)

</div>

AgentRobustBench measures how tool-use language agents handle failures that arise in deployment, from typos and ambiguous tool names to API timeouts. It pairs a robustness benchmark with **ToolRL-DR**, a reinforcement learning recipe that trains agents on varied tool-use trajectories.

## Run six examples

The walkthrough uses **Linux, Python 3.11 or 3.12, Git, and an NVIDIA GPU supported by vLLM**. Check [vLLM's installation requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for GPU and driver compatibility. vLLM automatically downloads the checkpoint from Hugging Face when the server first starts and reuses the local cache on later runs. Allow time and disk space for the first download.

A terminal is a window where you paste commands and press Enter. Run the following steps in order. Keep terminal 1 open while using terminal 2, on the **same machine**.

### 1. Download the project and install the environment

In **terminal 1**, check your tools:

```bash
python3 --version
git --version
nvidia-smi
```

`python3` should report Python 3.11 or 3.12, and `nvidia-smi` should list your GPU. If a command is missing, install that tool or use a configured GPU machine before continuing.

Download into your home directory, create a separate Python environment, and install the dependencies:

```bash
git clone --depth 1 https://github.com/WillChow66/AgentRobustBench.git ~/AgentRobustBench
cd ~/AgentRobustBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements.txt
```

If you already downloaded the project, open that directory and start at `python3 -m venv .venv`. The evaluation data is included in the repository. No separate dataset download is needed.

### 2. Start the model server

Still in **terminal 1**, run:

```bash
vllm serve willchow66/ToolRL-DR-Full-Qwen2.5-3B --port 8000 --max-model-len 8192
```

vLLM loads the model and waits for evaluation requests. Leave it running. When startup completes, open **terminal 2** on the same machine and run:

```bash
cd ~/AgentRobustBench
source .venv/bin/activate
python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://localhost:8000/v1/models'))['data'][0]['id'])"
```

The last command should print `willchow66/ToolRL-DR-Full-Qwen2.5-3B`. If you see a connection error, check terminal 1 and wait for the server to finish starting.

### 3. Generate six predictions

In **terminal 2**, paste this entire command block:

```bash
python scripts/run_eval.py \
    --model-name willchow66/ToolRL-DR-Full-Qwen2.5-3B \
    --endpoint http://localhost:8000/v1 \
    --data-dir hf_data/datasets/api_eval \
    --files clean.jsonl realistic_typos.jsonl \
    --limit 3 --temperature 0 --max-workers 1 \
    --output-dir results/quick-demo
```

This runs three clean examples and their three typo variants. The runner summary should show `processed=3` and `errors=0` for each file. It saves:

```text
results/quick-demo/
  clean.predictions.jsonl
  realistic_typos.predictions.jsonl
  _run_summary.json
```

Each predictions file contains one JSON record per example, including the model's output. If `errors` is nonzero, inspect the `error` field in those files and the server log before scoring.

### 4. Score and inspect the results

Still in **terminal 2**, run:

```bash
python leaderboard/scoring/eval_results.py \
    --predictions results/quick-demo \
    --output results/quick-demo/scores.json \
    --per-sample-output results/quick-demo/per_sample_scores.jsonl
```

The console summary should start with `n=6`. It also reports accuracy by benchmark and perturbation category. Open `results/quick-demo/scores.json` for aggregate scores and `per_sample_scores.jsonl` to inspect each example. These six examples check the workflow; use the full benchmark below for a paper-scale comparison. Scoring uses no model judge or external API.

To try ToolRL-DR-Mixed, replace the model ID in both terminals with `willchow66/ToolRL-DR-Mixed-Qwen2.5-3B` and use a new output directory such as `results/quick-demo-mixed`.

For a client on another computer, follow the [client-only setup](scripts/README.md#if-you-already-have-a-model-server). For all evaluation cases, continue to [full benchmark evaluation](scripts/README.md#full-benchmark-evaluation).

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
| `hf_data/datasets/api_eval/` | 17 JSONL files: 16 static perturbation cells + the clean baseline (3{,}721 predictions per model in total). The 6 Transition variants are produced at runtime by `scripts/run_eval.py`, not stored as separate files. |
| `scripts/` | Inference runner (`run_eval.py`), reparser (`reparse_predictions.py`), model manifest, and SLURM templates. See `scripts/README.md`. |
| `leaderboard/` | HuggingFace Space implementation: Gradio app, deterministic scorer, leaderboard CSV. See `leaderboard/README.md`. |
| `training/` | Reproduction recipe for ToolRL-DR-Full and ToolRL-DR-Mixed: the parquets actually fed to the GRPO trainer (`train.parquet`, `mixed_train.parquet`, `test.parquet`), the reference jsonl→parquet converter, our drop-in reward-score patch (`rlla_perturbed.py`), the GRPO launcher, and a hyperparameter manifest. See `training/README.md`. |

## Model checkpoints

| Checkpoint | Training data | Model card |
| --- | --- | --- |
| ToolRL-DR-Full | Perturbed trajectories | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Full-Qwen2.5-3B) |
| ToolRL-DR-Mixed | Clean and perturbed trajectories | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Mixed-Qwen2.5-3B) |

The checkpoints are based on Qwen2.5-3B-Instruct and use the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE). See [training/](training/) for the training data and reproduction instructions.

## Quick start

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

Code: Apache-2.0. Data: see per-source license documented in the paper.
