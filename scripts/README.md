# AgentRobustBench: Evaluation guide

[Project home](../README.md) · [中文入门指南](../README_zh-CN.md)

For direct CPU, Apple MPS, or CUDA inference without a model server, start with the [project quick start](../README.md#evaluate-the-full-benchmark). This page describes the optional vLLM server/client workflow.

## Evaluate the full benchmark with vLLM

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

### 3. Generate predictions for the complete benchmark

<a id="full-benchmark-evaluation"></a>

In **terminal 2**, run all static cases:

```bash
python scripts/run_eval.py \
    --model-name willchow66/ToolRL-DR-Full-Qwen2.5-3B \
    --endpoint http://localhost:8000/v1 \
    --data-dir hf_data/datasets/api_eval \
    --output-dir results/toolrl-dr-full \
    --mode single --temperature 0
```

Then run all six runtime error types:

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

Check each runner summary for `errors=0` before scoring. Predictions are saved
under `results/toolrl-dr-full/`, with runtime cases in `transition_*/`.

### 4. Score the complete benchmark

Still in **terminal 2**, run:

```bash
python leaderboard/scoring/eval_results.py \
    --predictions results/toolrl-dr-full \
    --output results/toolrl-dr-full/scores.json \
    --per-sample-output results/toolrl-dr-full/per_sample_scores.jsonl
```

A complete run processes **2,527 static cases** and **1,194 runtime cases**
(199 × 6), and the scorer should report **n=3721**. The runner skips 191
multi-turn records in the static files. The top-level accuracy includes clean
and perturbed cases; compare matching per-category metrics with the paper.

To evaluate ToolRL-DR-Mixed, use
`willchow66/ToolRL-DR-Mixed-Qwen2.5-3B` for the model server and all inference
commands, and use a separate output directory such as `results/toolrl-dr-mixed`
for inference and scoring.

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

In step 3, use your server's reachable URL ending in `/v1` for `--endpoint` and its registered model ID for `--model-name`. `localhost` means the computer running that command; use it only when the server is on the same computer. Pass `--api-key` if your endpoint requires authentication. Local vLLM uses the runner's default placeholder, so no OpenAI account or key is needed.

## Common problems

| What you see | What to do |
| --- | --- |
| `python` or `git` not found | Install the missing tool. Use `python3` on macOS/Linux or `py` on Windows to create the environment. |
| `No module named openai` or `tqdm` | Activate `.venv` and install `scripts/requirements-client.txt`. |
| `No such file` for a script | Run from the project directory, which contains `README.md`, `scripts/`, and `leaderboard/`. |
| Cannot connect to the endpoint | Keep the model server running, wait for startup, and check the URL using step 2. |
| Model not found by the server | Use the model ID printed by the step 2 check. |
| CUDA out of memory | Stop other GPU jobs or use a GPU with more free memory. See [vLLM memory guidance](https://docs.vllm.ai/en/latest/configuration/conserving_memory/). |
| `n` is not 3721 | Confirm that all static files and all six runtime error types completed, and score the correct output directory. |

## Files and cluster jobs

| Path | Role |
| --- | --- |
| `evaluate_checkpoint.py` | Directly loads Full/Mixed on CPU, MPS, or CUDA and scores the generated predictions. |
| `requirements-local.txt` | Dependencies for direct checkpoint inference. |
| `run_eval.py` | Reads benchmark JSONL files, sends model requests, and writes predictions. |
| `requirements-client.txt` | Dependencies for evaluating an existing model server. |
| `requirements.txt` | Client dependencies plus local vLLM model serving. |
| `reparse_predictions.py` | Re-runs tool-call parsing on saved predictions. |
| `model_manifest.py` | Model IDs and model-specific evaluation settings. |
| `slurm/` | Cluster job templates. Set your own allocation, filesystem paths, and scheduler settings before submission. |

See the [SLURM guide](slurm/README.md) for full-lineup evaluation and [training guide](../training/README.md) for retraining.
