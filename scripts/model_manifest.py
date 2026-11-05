"""
Model manifest for AgentRobustBench NeurIPS evaluation.

Each entry maps a model id (the ``--model-name`` we pass to run_eval.py) to
the vLLM server config needed to serve it. Consumed by the SLURM templates
under scripts/slurm/ and by helper scripts that spin up vLLM endpoints.

Conventions:
  - ``hf_name``: canonical HuggingFace repo id. Also the string we pass as
    ``--served-model-name`` to vLLM so the API request path matches.
  - ``gpus``: number of GPUs required for a single replica (tensor parallel).
  - ``dtype``: vLLM dtype. ToolRL-1.5B requires float32 (float16 produces
    inf/nan — see CLAUDE.md Critical Technical Notes).
  - ``max_model_len``: cap context to keep KV-cache in memory. Some newer
    models (Qwen3.5) claim >250K context by default which blows GPU memory.
  - ``extra_flags``: additional vLLM flags (e.g. MUA-RL's Hermes tool parser).
  - ``notes``: free text for operators.

Add new models by appending to ``MODELS``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional  # noqa: F401


@dataclass
class ModelConfig:
    hf_name: str
    gpus: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    extra_flags: List[str] = field(default_factory=list)
    category: str = ""           # for reporting only
    notes: str = ""
    # Qwen3 family switch. None → do not send the kwarg (legacy behavior).
    # False → pass enable_thinking=False per non-thinking-mode paper spec
    # (MUA-RL, LoopTool, Qwen3-*-Instruct all evaluate in non-thinking).
    enable_thinking: Optional[bool] = None
    # HF-card / paper-recommended sampling temp for this model. Only applied
    # if the SLURM submit script doesn't override it — per-model sampling
    # adjustments are needed e.g. for DeepSeek-R1-Distill (0.6).
    temperature: Optional[float] = None
    # Max generation tokens for this model (reasoning models need more
    # headroom to fit both think + answer under the context limit).
    max_tokens: Optional[int] = None
    # Function-calling mode: pass tools via OpenAI `tools=` API and rely on
    # vLLM's tool-call parser (hermes) to return structured tool_calls.
    # True for models trained/evaluated under FC mode (Qwen3 family —
    # MUA-RL, LoopTool, Qwen3, Qwen3.5). False for models trained in
    # prompt-mode (ToolRL, TL-CodeLLaMA, Qwen2.5-Instruct baseline for
    # ToolRL, SFT-Clean-4k, DeepSeek-R1-Distill).
    use_fc: bool = False


MODELS: Dict[str, ModelConfig] = {

    # ── Open-source tool-specialized RL / SFT models ───────────────
    # NOTE: max_model_len values reflect each model's HF config.json
    # max_position_embeddings. We often set lower than the HF max to
    # conserve KV-cache memory — our prompts are <=3K tokens and we cap
    # generation at max_tokens=1024 (4096 for reasoning models below).
    "chengq9/ToolRL-Qwen2.5-1.5B": ModelConfig(
        hf_name="chengq9/ToolRL-Qwen2.5-1.5B",
        gpus=1,
        dtype="float32",  # float16 causes inf/nan on this ckpt
        max_model_len=8192,   # HF max: 32768; 8K ample for our samples
        category="tool_rl_small",
        notes="Requires float32 per CLAUDE.md.",
    ),
    "chengq9/ToolRL-Qwen2.5-3B": ModelConfig(
        hf_name="chengq9/ToolRL-Qwen2.5-3B",
        gpus=1, dtype="bfloat16",
        max_model_len=8192,   # HF max: 32768
        category="tool_rl",
    ),
    "chengq9/ToolRL-Llama3.2-3B": ModelConfig(
        hf_name="chengq9/ToolRL-Llama3.2-3B",
        gpus=1, dtype="bfloat16",
        max_model_len=8192,   # HF max: 131072 (Llama 3.2 long-ctx)
        category="tool_rl",
    ),
    "Junjie-Ye/TL-CodeLLaMA-2": ModelConfig(
        hf_name="Junjie-Ye/TL-CodeLLaMA-2",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,   # HF max: 16384; full to be safe
        category="tool_sft",
        notes="Non-standard chat template (System:/User:/Assistant:/Output:); "
              "provide --chat-template scripts/slurm/tl_codellama_chat_template.jinja.",
    ),
    "zhuiguang-ning/LoopTool-8B": ModelConfig(
        hf_name="zhuiguang-ning/LoopTool-8B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,   # HF max: 40960
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="tool_rl",
        notes="Qwen3-8B-based; non-thinking for tool calling.",
    ),
    "zzwkk/MUA-RL-32B": ModelConfig(
        hf_name="zzwkk/MUA-RL-32B",
        gpus=1, dtype="bfloat16",  # 32B bf16 ~64GB, fits on 120GB GH200
        max_model_len=16384,   # HF max: 40960
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="tool_rl_large",
        notes="Qwen3-32B-based; paper evaluates in non-thinking mode at "
              "temperature=0.0 (arXiv 2508.18669 §4.1 Evaluation).",
    ),

    # ── MUA-RL family size sweep (arXiv 2508.18669) ───────────────
    "zzwkk/MUA-RL-8B": ModelConfig(
        hf_name="zzwkk/MUA-RL-8B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="tool_rl",
        notes="Qwen3-8B-based; non-thinking per MUA-RL paper.",
    ),
    "zzwkk/MUA-RL-14B": ModelConfig(
        hf_name="zzwkk/MUA-RL-14B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="tool_rl",
        notes="Qwen3-14B-based; non-thinking per MUA-RL paper.",
    ),

    # ── LoopTool family size sweep (arXiv 2511.09148) ─────────────
    "zhuiguang-ning/LoopTool-32B": ModelConfig(
        hf_name="zhuiguang-ning/LoopTool-32B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="tool_rl_large",
        notes="Qwen3-32B-based; paper uses FC mode with official BFCL-v3 "
              "eval script. Non-thinking assumed (HF card doesn't specify; "
              "tool-calling in Qwen3 recommends non-thinking).",
    ),

    # ── Qwen3 baselines for MUA-RL / LoopTool family ──────────────
    "Qwen/Qwen3-8B": ModelConfig(
        hf_name="Qwen/Qwen3-8B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,   # HF native: 32768
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="base_pair",
        notes="Base for MUA-RL-8B and LoopTool-8B. Non-thinking for "
              "tool-calling eval per Qwen3 model card.",
    ),
    "Qwen/Qwen3-14B": ModelConfig(
        hf_name="Qwen/Qwen3-14B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="base_pair",
        notes="Base for MUA-RL-14B. Non-thinking.",
    ),
    "Qwen/Qwen3-32B": ModelConfig(
        hf_name="Qwen/Qwen3-32B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="base_pair",
        notes="Base for MUA-RL-32B and LoopTool-32B. Non-thinking.",
    ),

    # ── Our proposed ToolRL-DR checkpoints ─────────────────────────
    # Backbone: Qwen2.5-3B-Instruct. Match the same eval mode used for the
    # public ToolRL-Qwen2.5-3B (prompt mode) so the comparison is apples-to-apples.
    # HF identifiers anonymized for double-blind review.
    "<anonymized>/qwen2.5-3b-toolrl-robustbench-perturbed": ModelConfig(
        hf_name="<anonymized>/qwen2.5-3b-toolrl-robustbench-perturbed",
        gpus=1, dtype="bfloat16", max_model_len=8192,
        category="proposed_rl",
        notes="ToolRL-DR-Full: RL on 100% perturbed trajectories.",
    ),
    "<anonymized>/qwen2.5-3b-toolrl-robustbench-mixed": ModelConfig(
        hf_name="<anonymized>/qwen2.5-3b-toolrl-robustbench-mixed",
        gpus=1, dtype="bfloat16", max_model_len=8192,
        category="proposed_rl",
        notes="ToolRL-DR-Mixed: RL on mixed_train.parquet "
              "(clean + perturbed mixture).",
    ),

    # ── Base pairs for RL-training ablation ────────────────────────
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig(
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        gpus=1, dtype="bfloat16",
        max_model_len=8192,   # HF max: 32768
        category="base_pair",
        notes="Base for ToolRL-Qwen2.5-1.5B comparison.",
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        gpus=1, dtype="bfloat16",
        max_model_len=8192,   # HF max: 32768
        category="base_pair",
        notes="Base for ToolRL-Qwen2.5-3B and ToolRL-Robust-3B comparison.",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelConfig(
        hf_name="meta-llama/Llama-3.2-3B-Instruct",
        gpus=1, dtype="bfloat16",
        max_model_len=8192,   # HF max: 131072
        category="base_pair",
        notes="Base for ToolRL-Llama3.2-3B comparison. Gated — "
              "HF_TOKEN required.",
    ),

    # ── Frontier open-weight models ────────────────────────────────
    "Qwen/Qwen3.5-9B": ModelConfig(
        hf_name="Qwen/Qwen3.5-9B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,   # HF claims 262K native; cap for KV-cache budget
        extra_flags=[
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        enable_thinking=False,
        use_fc=True,
        category="frontier_open",
        notes="Released 2026-02. Qwen3-family; needs non-thinking + hermes "
              "tool parser just like MUA-RL / LoopTool / Qwen3-Instruct.",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        gpus=1, dtype="bfloat16",
        max_model_len=16384,   # HF max: 131072
        category="frontier_open",
        notes="Reasoning model. Output includes <think>...</think> prefix — "
              "parser _strip_thinking() handles this. NOTE: default runner "
              "max_tokens=1024 is too small; the model spends many tokens "
              "thinking and gets truncated. Retry with MAX_TOKENS=4096.",
    ),
    # NOTE: Gemma-4 (2026-04) dropped from NeurIPS lineup: requires
    # transformers >= 5.0 for `gemma4` architecture, but vLLM 0.19
    # pins transformers < 5. Upgrading both together risks breaking
    # the other 12 models. Revisit when vLLM >= 0.20 ships with
    # transformers 5.x compatibility.
    #
    # "google/gemma-4-26B-A4B-it": ModelConfig(...)

    # ── Frontier closed-source (called via OpenAI API) ─────────────
    # Not served via local vLLM; called via the OpenAI Chat Completions API.
    "openai/o4-mini": ModelConfig(
        hf_name="openai/o4-mini",  # identifier only; not served locally
        gpus=0, dtype="",
        category="frontier_closed",
        notes="Closed-source; no local vLLM. "
              "Pass as --model-name o4-mini --endpoint https://api.openai.com/v1 "
              "with OPENAI_API_KEY set.",
    ),
}


def get_model_config(model_id: str) -> ModelConfig:
    """Look up model config by id. Raises KeyError if unknown."""
    if model_id not in MODELS:
        raise KeyError(
            f"Model '{model_id}' not in manifest. Known: {list(MODELS.keys())}"
        )
    return MODELS[model_id]


def models_by_category() -> Dict[str, List[str]]:
    """Group model ids by their category field — useful for paper tables."""
    out: Dict[str, List[str]] = {}
    for mid, cfg in MODELS.items():
        out.setdefault(cfg.category, []).append(mid)
    return out


if __name__ == "__main__":
    # Print manifest as a human-readable table.
    print(f"{'model_id':55s} {'gpus':4s} {'dtype':10s} {'category':20s} {'notes'}")
    print("-" * 150)
    for mid, cfg in MODELS.items():
        print(f"{mid:55s} {cfg.gpus:<4d} {cfg.dtype:10s} "
              f"{cfg.category:20s} {cfg.notes[:70]}")
