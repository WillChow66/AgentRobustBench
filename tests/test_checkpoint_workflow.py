"""CPU integration checks use a tiny local model, without downloading HF weights."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_checkpoint as workflow


class ConfigurationTests(unittest.TestCase):
    def test_released_checkpoint_ids(self):
        self.assertEqual(workflow.checkpoint_config("full")["model_id"],
                         "willchow66/ToolRL-DR-Full-Qwen2.5-3B")
        self.assertEqual(workflow.checkpoint_config("mixed")["model_id"],
                         "willchow66/ToolRL-DR-Mixed-Qwen2.5-3B")

    def test_cpu_auto_selection_and_unavailable_device(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        self.assertEqual(workflow.choose_device(torch, "auto"), "cpu")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            workflow.choose_device(torch, "cuda")

    def test_help_needs_no_model_download(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_checkpoint.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device", result.stdout)


class WorkflowTests(unittest.TestCase):
    def test_default_run_scores_the_entire_benchmark(self):
        # Exercise real dataset loading, all static/runtime jobs, and the scorer.
        # Only model generation is replaced, to keep this a CPU-only CI check.
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="[]", tool_calls=None))])
        )))
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "output"
            with patch("transformers.AutoTokenizer.from_pretrained"), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained"), \
                 patch.object(workflow, "LocalModelClient", return_value=client):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = workflow.main(["--device", "cpu", "--output-dir", str(output)])
            self.assertEqual(result, 0)
            scores = json.loads((output / "scores.json").read_text())
            self.assertEqual(scores["overall"]["overall"]["n"], 3721)
            self.assertEqual(scores["overall"]["by_mdp_category"]["clean"]["n"], 199)
            self.assertEqual(scores["overall"]["by_mdp_category"]["transition"]["n"], 1194)
            self.assertEqual(len(list(output.glob("transition_*"))), 6)
            self.assertTrue((output / "comparison.csv").is_file())
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["suite"], "full")

    def test_endpoint_runner_exits_nonzero_after_request_errors(self):
        import run_eval
        def fail(**kwargs):
            raise RuntimeError("test endpoint failure")
        client = SimpleNamespace(models=SimpleNamespace(list=lambda: []),
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with tempfile.TemporaryDirectory() as td:
            argv = ["run_eval.py", "--model-name", "test-model", "--files", "clean.jsonl",
                    "--limit", "1", "--output-dir", td]
            with patch.object(run_eval, "OpenAI", return_value=client), patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        run_eval.main()
            self.assertEqual(raised.exception.code, 1)
            summary = json.loads((Path(td) / "_run_summary.json").read_text())
            self.assertEqual(summary["files"][0]["errors"], 1)

    def test_comparison_uses_newly_measured_scores(self):
        categories = {k: {"n": 2, "correct": 1, "accuracy": 0.5}
                      for k in ("clean", "observation", "action", "reward", "transition")}
        with tempfile.TemporaryDirectory() as td:
            path = workflow.write_comparison({"overall": {"by_mdp_category": categories}},
                                             workflow.checkpoint_config("full"), Path(td))
            import csv
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(float(row["measured"]), 0.5)
                self.assertAlmostEqual(float(row["difference_pp"]),
                                       100 * (0.5 - float(row["released"])))

    def test_generation_failure_stops_before_scoring(self):
        import run_eval
        def fail(**kwargs):
            raise RuntimeError("test model failure")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "generation errors"), contextlib.redirect_stdout(io.StringIO()):
                workflow.evaluate(run_eval, client, workflow.checkpoint_config("full"), "demo", Path(td), 4)
            self.assertFalse((Path(td) / "scores.json").exists())

    def test_six_case_real_transformers_cpu_generation(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        with tempfile.TemporaryDirectory() as td:
            model_dir, output = Path(td) / "model", Path(td) / "output"
            model_dir.mkdir()
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
                                           "user": 3, "assistant": 4, "system": 5,
                                           "hello": 6, "tool": 7}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
                "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            )
            tokenizer.save_pretrained(model_dir)
            config = Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                 max_position_embeddings=16384, pad_token_id=1, eos_token_id=2)
            torch.manual_seed(42)
            Qwen2ForCausalLM(config).save_pretrained(model_dir)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate_checkpoint.py"),
                "--device", "cpu", "--model-path", str(model_dir), "--suite", "demo",
                "--max-tokens", "4", "--max-context", "16384", "--cpu-threads", "1",
                "--output-dir", str(output),
            ], text=True, capture_output=True, timeout=120)
            prediction_errors = []
            for path in output.rglob("*.predictions.jsonl"):
                prediction_errors.extend(json.loads(line)["prediction"].get("error")
                                         for line in path.read_text().splitlines())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + str(prediction_errors))
            scores = json.loads((output / "scores.json").read_text())
            self.assertEqual(scores["overall"]["overall"]["n"], 6)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["model_source"], str(model_dir))
            self.assertIn("evaluation_seconds", metadata)
            # Reusing a populated directory must fail before loading a model.
            again = subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate_checkpoint.py"),
                "--device", "cpu", "--model-path", str(model_dir), "--output-dir", str(output),
            ], text=True, capture_output=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("not empty", again.stderr)


if __name__ == "__main__":
    unittest.main()
