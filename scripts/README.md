# AgentRobustBench: Evaluation guide

[Project home](../README.md) · [中文入门指南](../README_zh-CN.md)

The [browser demo](https://colab.research.google.com/github/WillChow66/AgentRobustBench/blob/main/docs/quickstart.ipynb) recomputes a published result on CPU. This guide shows how to run a model, generate new predictions, and score them.

## Run six examples

The walkthrough uses **Linux, Python 3.11 or 3.12, Git, and an NVIDIA GPU supported by vLLM**. Check [vLLM's installation requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for GPU and driver compatibility. The checkpoint is downloaded from Hugging Face when the server first starts, so allow time and disk space for that download.

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

### If you already have a model server

You can run the evaluation client on another computer, including macOS or Windows. Install Python 3.11 or 3.12 and Git, clone the project, then create the client environment.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-client.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-client.txt
```

On Windows, replace `python` in the evaluation commands with `.\.venv\Scripts\python.exe` and put each command on one line, removing the trailing `\` line continuations.

In steps 3–4, use your server's reachable URL ending in `/v1` for `--endpoint` and its registered model ID for `--model-name`. `localhost` means the computer running that command; use it only when the server is on the same computer. Pass `--api-key` if your endpoint requires authentication. Local vLLM uses the runner's default placeholder, so no OpenAI account or key is needed.

## Full benchmark evaluation

With the server running and the environment active, run the following in **terminal 2** from the project directory. First evaluate all static splits:

```bash
python scripts/run_eval.py \
    --model-name willchow66/ToolRL-DR-Full-Qwen2.5-3B \
    --endpoint http://localhost:8000/v1 \
    --data-dir hf_data/datasets/api_eval \
    --output-dir results/toolrl-dr-full \
    --mode single --temperature 0
```

Then run the six runtime perturbations. This loop is for Linux or macOS terminals:

```bash
for T in timeout rate_limit auth_error server_error malformed_response schema_drift; do
  python scripts/run_eval.py \
      --model-name willchow66/ToolRL-DR-Full-Qwen2.5-3B \
      --endpoint http://localhost:8000/v1 \
      --data-dir hf_data/datasets/api_eval \
      --output-dir results/toolrl-dr-full \
      --mode transition --transition-type "$T" --temperature 0
done
```

Finally, score the entire output directory, including the `transition_*` subdirectories:

```bash
python leaderboard/scoring/eval_results.py \
    --predictions results/toolrl-dr-full \
    --output results/toolrl-dr-full/scores.json \
    --per-sample-output results/toolrl-dr-full/per_sample_scores.jsonl
```

A complete run processes **2,527 static cases** and **1,194 runtime cases** (199 × 6), so the scorer should report **n=3721**. The runner skips 191 multi-turn records present in the static files.

Use a separate output directory per model and per experiment. `--resume` continues an interrupted run by skipping saved IDs; retry failed predictions in a new directory instead. Generation settings and model-specific flags are described in [`model_manifest.py`](model_manifest.py) and the [SLURM guide](slurm/README.md).

## Common problems

| What you see | What to do |
| --- | --- |
| `python` or `git` not found | Install the missing tool. Use `python3` on macOS/Linux or `py` on Windows to create the environment. |
| `No module named openai` or `tqdm` | Activate `.venv` and install `scripts/requirements-client.txt`. |
| `No such file` for a script | Run from the project directory, which contains `README.md`, `scripts/`, and `leaderboard/`. |
| Cannot connect to the endpoint | Keep the model server running, wait for startup, and check the URL using step 2. |
| Model not found by the server | Use the model ID printed by the step 2 check. |
| CUDA out of memory | Stop other GPU jobs or use a GPU with more free memory. See [vLLM memory guidance](https://docs.vllm.ai/en/latest/configuration/conserving_memory/). |
| `n` is not 6 in the small example | Use a new output directory containing only this six-example run, and check both files' runner summaries. |

## Files and cluster jobs

| Path | Role |
| --- | --- |
| `run_eval.py` | Reads benchmark JSONL files, sends model requests, and writes predictions. |
| `reproduce_transition_gap.py` | Recomputes the published Transition gap reduction from `leaderboard/leaderboard.csv` using Python alone. |
| `requirements-client.txt` | Dependencies for evaluating an existing model server. |
| `requirements.txt` | Client dependencies plus local vLLM model serving. |
| `reparse_predictions.py` | Re-runs tool-call parsing on saved predictions. |
| `model_manifest.py` | Model IDs and model-specific evaluation settings. |
| `slurm/` | Cluster job templates. Set your own allocation, filesystem paths, and scheduler settings before submission. |

See the [SLURM guide](slurm/README.md) for full-lineup evaluation and [training guide](../training/README.md) for retraining.
