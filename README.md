<div align="center">

# AgentRobustBench

**English** | [简体中文](README_zh-CN.md)

**A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents**

NeurIPS 2026 · Evaluations & Datasets Track

[![Paper on arXiv](https://img.shields.io/badge/arXiv-2605.11928-b31b1b)](https://arxiv.org/abs/2605.11928)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench)
[![Live leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-7C3AED)](https://huggingface.co/spaces/willchow66/AgentRobustBench)

**199 seed tasks · 22 perturbation types · 3,721 evaluation cases · 21 models evaluated**

[Five-minute reproduction](#reproduce-a-paper-result-in-five-minutes) · [Benchmark](#benchmark) · [Results](#results) · [Full evaluation](#quick-start) · [Checkpoints](#model-checkpoints) · [Training](#reproducing-toolrl-dr-optional) · [Citation](#citation)

</div>

AgentRobustBench measures how tool-use language agents handle failures that arise in deployment, from typos and ambiguous tool names to API timeouts. It pairs a robustness benchmark with **ToolRL-DR**, a reinforcement learning recipe that trains agents on varied tool-use trajectories.

## Reproduce a paper result in five minutes

New to Python? Start with the browser demo below. It recomputes the paper's finding that ToolRL-DR-Full closes about 27% of the 3B ToolRL baseline's Transition gap, using the released leaderboard scores.

### Option 1: Run in your browser — no installation or GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillChow66/AgentRobustBench/blob/main/docs/quickstart.ipynb)

1. Click **Open in Colab**. Sign in to Google if prompted.
2. Keep the default **CPU** runtime. Select **Runtime → Run all** and allow execution of this repository's notebook if Colab asks.
3. Read the output below the code cell. You should see:

```text
ToolRL-Clean Transition gap:    31.49 pp
ToolRL-DR-Full Transition gap:  23.11 pp
Gap reduction:                  8.38 pp
Baseline gap closed:            26.6%
```

Here, `pp` means percentage points. The script subtracts Transition accuracy from clean accuracy for each model, then divides the gap reduction by the baseline gap. The notebook downloads only the result calculator and the small score table.

### Option 2: Run the same calculation on your computer

You need [Git](https://git-scm.com/downloads) and [Python 3](https://www.python.org/downloads/). On macOS, open **Terminal**; on Windows, open **PowerShell**. Paste one command at a time, pressing Enter after each:

```bash
git clone --depth 1 https://github.com/WillChow66/AgentRobustBench.git
cd AgentRobustBench
python3 scripts/reproduce_transition_gap.py
```

On Windows, use `py scripts/reproduce_transition_gap.py` for the last command. The output should match the four lines above. This calculation needs no Python packages, model download, API key, or GPU.

### Run the model on six examples

To generate new predictions, follow the [step-by-step evaluation guide](scripts/README.md#run-six-examples). It covers environment installation, starting the released checkpoint, running three clean and three typo examples, and checking the scores. The walkthrough uses a Linux machine with an NVIDIA GPU and Python 3.11 or 3.12. A Mac or Windows computer can also send requests to an existing model server using the client-only setup in that guide.

After a successful six-example run, the scorer prints `n=6` and writes `results/quick-demo/scores.json`. Model installation, the first checkpoint download, and full benchmark evaluation take additional time. The browser demo above is the five-minute entry point for checking the published calculation.

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
