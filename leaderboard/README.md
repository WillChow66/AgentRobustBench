---
title: RobustBench-TC Leaderboard
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

# RobustBench-TC Leaderboard (MVP)

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

The expected `sample_id` and `perturbation` values are documented in our HF dataset `<anonymized>/RobustBench-TC`.

## Local-eval CLI

We provide a CLI that runs the same scorer locally so users can validate before submitting:

```bash
pip install robustbench-tc
robustbench-tc score --predictions my_predictions.json --output scores.json
```

The CLI is a thin wrapper around `scripts/eval_results.py`.

## Status

- [ ] M1: Gradio app skeleton (this directory)
- [ ] M2: Submission validator (schema + sample-ID coverage check)
- [ ] M3: Hosted scorer reusing `scripts/eval_results.py`
- [ ] M4: Persistence to HF Dataset
- [ ] M5: Leaderboard rendering (sorted by overall acc, with per-MDP-component drop columns)
- [ ] M6: Deployed on HF Space `<anonymized>/robustbench-tc-leaderboard`
- [ ] M7: Documentation + first submission with our 21 evaluated models

ETA for M1--M5: 1 week eng. M6 + M7 another 2 days.
