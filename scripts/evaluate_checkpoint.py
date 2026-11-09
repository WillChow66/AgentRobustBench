#!/usr/bin/env python3
"""Load a released checkpoint, generate predictions, and score them locally."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LOCAL_REQUIREMENTS = ROOT / "scripts" / "requirements-local.txt"


def checkpoint_config(name, path=ROOT / "configs" / "checkpoints.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))[name]


def choose_device(torch, requested):
    """Select an available device; an explicit choice never falls back silently."""
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if available[d])
    if not available[requested]:
        raise ValueError(f"Device {requested!r} is unavailable. Use --device cpu or --device auto.")
    return requested


def package_versions():
    result = {}
    for name in ("torch", "transformers", "accelerate", "openai", "tqdm", "vllm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class LocalModelClient:
    """Adapt Transformers generation to the existing prompt/parser pipeline."""

    def __init__(self, model, tokenizer, torch, device, context_length):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = device
        self.context_length = context_length
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, max_tokens=1024, temperature=0, **kwargs):
        if kwargs.get("tools"):
            raise ValueError("Local checkpoint inference uses the benchmark's prompt-based tool protocol.")
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        # Qwen2 accepts token IDs and attention masks, not segment/type IDs.
        inputs.pop("token_type_ids", None)
        prompt_length = inputs["input_ids"].shape[-1]
        if prompt_length + max_tokens > self.context_length:
            raise ValueError(
                f"Prompt ({prompt_length}) plus max tokens ({max_tokens}) exceeds "
                f"--max-context {self.context_length}. Increase --max-context."
            )
        generation = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation["temperature"] = temperature
        # Disable sampling defaults inherited from the checkpoint for greedy decoding.
        else:
            generation.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        text = self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        message = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def write_comparison(summary, checkpoint, output_dir):
    """Compare newly generated full-run metrics with the released score table."""
    categories = summary["overall"]["by_mdp_category"]
    perturbed = [v for k, v in categories.items() if k != "clean"]
    metrics = {
        "Pert. Acc.": sum(v["correct"] for v in perturbed) / sum(v["n"] for v in perturbed),
        "Clean": categories["clean"]["accuracy"],
        "Obs Acc.": categories["observation"]["accuracy"],
        "Action Acc.": categories["action"]["accuracy"],
        "Reward Acc.": categories["reward"]["accuracy"],
        "Transition Acc.": categories["transition"]["accuracy"],
    }
    with (ROOT / "leaderboard" / "leaderboard.csv").open(newline="", encoding="utf-8") as f:
        reference = next(r for r in csv.DictReader(f) if r["Model"] == checkpoint["leaderboard_name"])
    path = output_dir / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "measured", "released", "difference_pp"])
        writer.writeheader()
        for metric, value in metrics.items():
            released = float(reference[metric])
            writer.writerow({"metric": metric, "measured": value, "released": released,
                             "difference_pp": 100 * (value - released)})
    return path


def evaluate(runner, client, checkpoint, suite, output_dir, max_tokens):
    """Use the existing benchmark runner and scorer, without an HTTP server."""
    data_dir = ROOT / "hf_data" / "datasets" / "api_eval"
    if suite == "demo":
        jobs = [(data_dir / name, "single", None)
                for name in ("clean.jsonl", "realistic_typos.jsonl")]
    else:
        jobs = [(p, "single", None) for p in sorted(data_dir.glob("*.jsonl"))]
        jobs.extend((data_dir / "clean.jsonl", "transition", kind)
                    for kind in runner.TRANSITION_ERROR_MESSAGES)
    counts = []
    for path, mode, kind in jobs:
        target = output_dir / (f"transition_{kind}" if kind else "") / f"{path.stem}.predictions.jsonl"
        print(f"Running {path.name}" + (f" / {kind}" if kind else ""), flush=True)
        processed, skipped, errors = runner.run_file(
            path, target, client, model_name=checkpoint["model_id"],
            temperature=0, max_tokens=max_tokens, use_fc=False,
            max_workers=1, skip_multi_turn=True, resume=False, mode=mode,
            transition_error_message=runner.TRANSITION_ERROR_MESSAGES[kind] if kind else None,
            limit=3 if suite == "demo" else None,
        )
        counts.append({"file": str(target.relative_to(output_dir)), "processed": processed,
                       "skipped_multi_turn": skipped, "errors": errors})
        (output_dir / "_run_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
        if errors:
            raise RuntimeError(f"{errors} model generation errors in {target}. Check its error fields.")
    subprocess.run([
        sys.executable, str(ROOT / "leaderboard" / "scoring" / "eval_results.py"),
        "--predictions", str(output_dir), "--output", str(output_dir / "scores.json"),
        "--per-sample-output", str(output_dir / "per_sample_scores.jsonl"),
    ], check=True)
    summary = json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))
    expected = 6 if suite == "demo" else 3721
    actual = summary["overall"]["overall"]["n"]
    if actual != expected:
        raise RuntimeError(f"Expected {expected} scored examples, got {actual}. Check the dataset and run summary.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("full", "mixed"), default="full")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--suite", choices=("demo", "full"), default="demo")
    parser.add_argument("--check", action="store_true", help="Check installed dependencies and device; do not download a model.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, help="Load an existing local model/tokenizer directory instead of HF.")
    args = parser.parse_args(argv)
    if min(args.max_tokens, args.max_context, args.cpu_threads) < 1:
        parser.error("Token limits and CPU thread count must be positive.")
    config = checkpoint_config(args.checkpoint)
    versions = package_versions()
    missing = [p for p in ("torch", "transformers", "accelerate", "openai", "tqdm") if versions[p] is None]
    print(f"Computer: {platform.system()} / {platform.machine()}")
    print(f"Checkpoint: {config['model_id']}")
    if missing:
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        print(f"Install with: {sys.executable} -m pip install -r {LOCAL_REQUIREMENTS}", file=sys.stderr)
        return 1
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        import run_eval as runner
        device = choose_device(torch, args.device)
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        )
        print(f"Device: {device}; dtype: {dtype}")
        print("Packages: " + json.dumps(versions))
        import psutil
        available_ram = psutil.virtual_memory().available
        print(f"Available system RAM: {available_ram / 2**30:.1f} GiB")
        if device == "cpu":
            print("The released checkpoint has about 13.6 GB of float32 weights, plus inference memory.")
        if args.check:
            print("Environment check passed. No model weights downloaded.")
            return 0
        data_dir = ROOT / "hf_data" / "datasets" / "api_eval"
        if not (data_dir / "clean.jsonl").is_file() or not (data_dir / "realistic_typos.jsonl").is_file():
            raise ValueError(f"Evaluation data not found in {data_dir}. Download the complete repository.")
        output = args.output_dir or ROOT / "results" / f"{args.checkpoint}-{args.suite}-{time.time_ns()}"
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}. Choose a new directory.")
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "checkpoint": config, "model_source": str(args.model_path) if args.model_path else config["model_id"],
            "suite": args.suite, "device": device, "dtype": str(dtype), "python": sys.version,
            "platform": platform.platform(), "packages": versions, "seed": args.seed,
            "temperature": 0, "max_tokens": args.max_tokens, "max_context": args.max_context,
            "cpu_threads": args.cpu_threads, "status": "starting",
            "available_ram_bytes_at_start": available_ram,
            "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.jsonl"))},
        }
        if device == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name(0)
        try:
            metadata["code_revision"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["code_revision"] = None
        manifest = output / "run_metadata.json"
        def save():
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        save()
        started = time.monotonic()
        try:
            set_seed(args.seed)
            torch.set_num_threads(args.cpu_threads)
            source = str(args.model_path) if args.model_path else config["model_id"]
            revision = {} if args.model_path else {"revision": config["revision"]}
            print("Loading tokenizer and checkpoint. The first run downloads the HF weights.", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(source, **revision)
            model = AutoModelForCausalLM.from_pretrained(
                source, torch_dtype=dtype, device_map={"": device},
                use_safetensors=True, **revision,
            ).eval()
            client = LocalModelClient(model, tokenizer, torch, device, args.max_context)
            metadata["load_seconds"] = time.monotonic() - started
            inference_start = time.monotonic()
            summary = evaluate(runner, client, config, args.suite, output, args.max_tokens)
            metadata["evaluation_seconds"] = time.monotonic() - inference_start
            metadata["scored_examples"] = summary["overall"]["overall"]["n"]
            if args.suite == "full" and not args.model_path:
                print(f"Measured vs released scores: {write_comparison(summary, config, output)}")
            metadata["status"] = "completed"
            print(f"Results: {output.resolve()}")
        except BaseException:
            metadata["status"] = "failed"
            raise
        finally:
            metadata["total_seconds"] = time.monotonic() - started
            save()
        return 0
    except KeyboardInterrupt:
        print("Stopped. Partial predictions remain in the output directory.", file=sys.stderr)
        return 130
    except (Exception, SystemExit) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
