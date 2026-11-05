---
title: AgentRobustBench Leaderboard
emoji: 🛠️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sim-to-real robustness leaderboard for tool-use LLM agents
python_version: "3.11"
---

# AgentRobustBench Leaderboard

> A BFCL-style live leaderboard for tool-use sim-to-real robustness, deployed as a HuggingFace Space.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│ User submits        │  POST   │ Gradio backend (HF   │
│ predictions JSON    ├────────►│ Space, Python)       │
│ (via UI or CLI)     │         │ • Validates schema   │
└─────────────────────┘         │ • Runs eval_results  │
                                │   in subprocess      │
                                │ • Computes per-MDP   │
                                │   accuracy + drop    │
                                │ • Persists to HF     │
                                │   Dataset hub        │
                                └──────────┬───────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       │                                       │
                       ▼                                       ▼
            ┌───────────────────┐                ┌──────────────────────┐
            │ HF Dataset:       │                │ Public leaderboard   │
            │ submissions log   │                │ table, refreshed on  │
            │ (append-only)     │                │ each submission      │
            └───────────────────┘                └──────────────────────┘
```

## Submission schema

A valid submission is a JSON file with this shape:

```json
{
  "model_name": "MyTeam/MyModel-7B",
  "model_type": "open" | "closed-api" | "fine-tuned",
  "contact": "user@example.com",
  "results": [
    {
      "sample_id": "bfcl_v3__BFCL_v3_simple__1",
      "perturbation": "Typo",
      "tool_calls": [{"name": "...", "parameters": {"...": "..."}}]
    },
    ...
  ]
}
```

The expected `sample_id` and `perturbation` values are documented in our HF dataset `robustbench-tc/RobustBench-TC`.

## Score locally

Run the scorer from the [code repository](https://github.com/WillChow66/AgentRobustBench) using a directory of prediction JSONL files:

```bash
python leaderboard/scoring/eval_results.py --predictions results/<model> --output scores.json
```

## Hosted leaderboard

[Open the leaderboard](https://huggingface.co/spaces/willchow66/AgentRobustBench) to submit predictions.
