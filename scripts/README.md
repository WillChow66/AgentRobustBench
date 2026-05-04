# Scripts Directory

This directory contains training, evaluation, and utility scripts for the Robust Tool Calling System.

---

## 📂 Script Categories

### ✅ Core Scripts (In Active Use)

#### 🏋️ Training Scripts
| Script | Description | Usage | Status |
|--------|-------------|-------|--------|
| `run_sft_training.py` | Supervised Fine-Tuning (SFT) implementation | `sbatch run_sft_train.slurm` | ✅ 61% win rate |
| `run_baseline_avatar.py` | AVATAR instruction optimization | `sbatch run_baseline_avatar_fixed.slurm` | ✅ 82% win rate (Best) |

#### 📊 Evaluation Scripts
| Script | Description | Purpose | Output |
|--------|-------------|---------|--------|
| `evaluate_perturbation_difficulty.py` | Input perturbation difficulty evaluation | Test baseline robustness | `results/baseline_perturbation_difficulty_eval.json` |
| `evaluate_negative_samples_comprehensive.py` | Output error detection evaluation | Test error handling | `results/negative_samples_comprehensive_eval_full.json` |
| `evaluate_sft_vs_baseline.py` | Two-way comparison: SFT vs Baseline | Head-to-head evaluation | `results/sft_vs_baseline_evaluation.json` |
| `evaluate_avatar_vs_baseline.py` | Two-way comparison: AVATAR vs Baseline | Head-to-head evaluation | `results/avatar_vs_baseline_evaluation.json` |
| `evaluate_three_models.py` | Four-way comparison (all models) | Overall ranking | `results/four_model_comparison.json` |
| `comprehensive_model_comparison.py` | Complete evaluation framework | Multi-model comparison | Various result files |

#### 🔍 Analysis & Utility Scripts
| Script | Description | Usage | Features |
|--------|-------------|-------|----------|
| `view_perturbation_results.py` | Interactive perturbation result viewer | `python scripts/view_perturbation_results.py results/...` | Filtering, comparison, stats |
| `view_eval_results.py` | Evaluation result viewer | `python scripts/view_eval_results.py results/...` | Interactive analysis |
| `check_data_format.py` | Data format validator | `python scripts/check_data_format.py data/...` | Format checking, stats |
| `validate_and_clean_data.py` | Data cleaning utility | `python scripts/validate_and_clean_data.py ...` | Remove invalid samples |
| `full_pipeline.py` | Complete training+evaluation pipeline | `python scripts/full_pipeline.py` | End-to-end automation |

---

## 📝 Detailed Script Descriptions

### Training Scripts

#### `run_sft_training.py`
**Purpose**: Supervised Fine-Tuning (SFT) on perturbed training samples

**What it does**:
- Loads Qwen2.5-1.5B-Instruct base model
- Fine-tunes on 2,000 samples (100 × 20 variants)
- Trains all 1.5B parameters (full fine-tuning)
- Saves checkpoints every epoch

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_sft_train.slurm

# Direct execution
python scripts/run_sft_training.py \
    --config configs/config_parl.yaml \
    --output_dir checkpoints/sft_model \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-5
```

**Parameters**:
- `--config`: Path to config file (default: `configs/config_parl.yaml`)
- `--output_dir`: Checkpoint save directory
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.01)
- `--max_length`: Maximum sequence length (default: 2048)

**Output**:
- Checkpoint: `<repo-root>/checkpoints/checkpoints/sft_robust_{job_id}/`
- Logs: `logs/sft_train_{job_id}.out`
- Training curves: WandB dashboard

**Performance**:
- Training time: ~1 hour on H200 GPU
- Win rate: 61% vs Baseline 24%
- Memory: ~6GB model + activations

---

#### `run_baseline_avatar.py`
**Purpose**: AVATAR instruction optimization without model training

**What it does**:
- Loads base model (no fine-tuning)
- Optimizes instruction prompts using GPT-4
- Evaluates instruction variants on validation set
- Selects best instruction based on performance

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_baseline_avatar_fixed.slurm

# Direct execution
python scripts/run_baseline_avatar.py \
    --config configs/config_parl.yaml \
    --output_dir checkpoints/avatar_model \
    --num_iterations 10 \
    --population_size 5
```

**Parameters**:
- `--config`: Path to config file
- `--output_dir`: Checkpoint save directory
- `--num_iterations`: Number of optimization iterations (default: 10)
- `--population_size`: Number of instruction candidates (default: 5)
- `--temperature`: GPT-4 temperature for generation (default: 0.7)

**Output**:
- Checkpoint: `checkpoints/phase2_iter_0.json` (optimized instructions)
- Logs: `logs/baseline_avatar_{job_id}.out`
- Best instruction: Stored in checkpoint JSON

**Performance**:
- Training time: ~30 minutes (no model training)
- Win rate: **82% vs Baseline 9%** (Best)
- Memory: Minimal (only inference)

---

### Evaluation Scripts

#### `evaluate_perturbation_difficulty.py`
**Purpose**: Evaluate model robustness to input perturbations of varying difficulty

**What it does**:
1. Loads test data with perturbation variants
2. Evaluates model on original and perturbed inputs
3. Compares accuracy across difficulty levels (easy, medium, hard)
4. Analyzes perturbation effects and difficulty gradient

**Usage**:
```bash
# Via SLURM (recommended for 100 samples)
sbatch run_baseline_perturbation_eval.slurm

# Direct execution (for small batches)
python scripts/evaluate_perturbation_difficulty.py \
    --model_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --test_data data/training_data_100.json \
    --output results/baseline_perturbation_difficulty_eval.json \
    --batch_size 1
```

**Parameters**:
- `--model_path`: Model path or HuggingFace ID
- `--test_data`: Path to test data with variants
- `--output`: Output JSON file path
- `--batch_size`: Evaluation batch size (default: 1)
- `--use_float32`: Use float32 instead of float16 (default: True)

**Output**:
```json
{
  "summary": {
    "total_samples": 1300,
    "original_accuracy": 0.23,
    "easy_accuracy": 0.23,
    "medium_accuracy": 0.2025,
    "hard_accuracy": 0.22
  },
  "per_variant": {
    "negative_1": {"correct": 23, "total": 100, "accuracy": 0.23},
    "negative_5": {"correct": 16, "total": 100, "accuracy": 0.16}
  },
  "samples": [...]
}
```

**Analysis**:
```bash
# View results with comparison
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --compare

# View most challenging perturbations
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --category negative_medium \
    --incorrect-only \
    --limit 10
```

**Expected Runtime**: ~7.5 hours (100 samples × 13 variants × 270s/sample)

---

#### `evaluate_negative_samples_comprehensive.py`
**Purpose**: Comprehensive evaluation of error detection capabilities

**What it does**:
1. Generates negative samples with various output errors
2. Evaluates 4 models: Baseline, SFT, LoRA, AVATAR
3. Tests error detection across difficulty levels
4. Compares error handling robustness

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_negative_eval.slurm

# Direct execution
python scripts/evaluate_negative_samples_comprehensive.py \
    --config configs/config_parl.yaml \
    --test_data data/test_data_100_enhanced.json \
    --output results/negative_samples_comprehensive_eval_full.json \
    --num_negative_variants 7
```

**Parameters**:
- `--config`: Configuration file
- `--test_data`: Test data path
- `--output`: Output JSON path
- `--num_negative_variants`: Number of negative variants per sample (default: 7)
- `--models`: Comma-separated model list (default: "baseline,sft,lora,avatar")

**Output**:
- Full results: `results/negative_samples_comprehensive_eval_full.json`
- Summary: `results/negative_samples_comprehensive_eval_full_summary.json`

**Analysis**:
```bash
# View results interactively
python scripts/view_eval_results.py \
    results/negative_samples_comprehensive_eval_full.json

# Filter by model and difficulty
python scripts/view_eval_results.py \
    results/negative_samples_comprehensive_eval_full.json \
    --model sft \
    --difficulty hard \
    --incorrect-only
```

**Expected Runtime**: ~12 hours (4 models × 100 samples × 7 variants)

---

#### `evaluate_sft_vs_baseline.py`
**Purpose**: Head-to-head comparison of SFT model vs Baseline

**What it does**:
1. Loads both SFT and Baseline models
2. Generates predictions for 100 test samples
3. Uses GPT-4.1-mini judge for pairwise comparison
4. Calculates win/loss/tie statistics

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_sft_vs_baseline_eval.slurm

# Direct execution
python scripts/evaluate_sft_vs_baseline.py \
    --config configs/config_parl.yaml \
    --test_data data/test_data_100_enhanced.json \
    --output results/sft_vs_baseline_evaluation.json \
    --judge_model "gpt-4.1-mini"
```

**Parameters**:
- `--config`: Configuration file
- `--test_data`: Test data path
- `--output`: Output JSON path
- `--judge_model`: GPT model for judging (default: "gpt-4.1-mini")
- `--sft_checkpoint`: SFT model checkpoint path

**Output**:
```json
{
  "summary": {
    "total_samples": 100,
    "sft_wins": 61,
    "baseline_wins": 24,
    "ties": 13,
    "sft_advantage": 0.37
  },
  "samples": [...]
}
```

**Expected Runtime**: ~2 hours (100 samples × 2 models + GPT-4 judging)

---

#### `evaluate_avatar_vs_baseline.py`
**Purpose**: Head-to-head comparison of AVATAR vs Baseline

**What it does**:
1. Loads AVATAR instructions and Baseline model
2. Generates predictions with optimized instructions
3. Uses GPT-4.1-mini judge for comparison
4. Calculates win/loss/tie statistics

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_avatar_vs_baseline_eval.slurm

# Direct execution
python scripts/evaluate_avatar_vs_baseline.py \
    --config configs/config_parl.yaml \
    --test_data data/test_data_100_enhanced.json \
    --output results/avatar_vs_baseline_evaluation.json \
    --avatar_checkpoint checkpoints/phase2_iter_0.json
```

**Output**:
```json
{
  "summary": {
    "total_samples": 100,
    "avatar_wins": 82,
    "baseline_wins": 9,
    "ties": 7,
    "avatar_advantage": 0.73
  }
}
```

**Expected Runtime**: ~1.5 hours (100 samples + GPT-4 judging)

---

#### `evaluate_three_models.py`
**Purpose**: Four-way simultaneous comparison of all models

**What it does**:
1. Loads all 4 models: Baseline, SFT, LoRA, AVATAR
2. Generates predictions from all models
3. Uses GPT-4.1-mini for four-way ranking
4. Calculates 1st/2nd/3rd/4th place statistics

**Usage**:
```bash
# Via SLURM (recommended)
sbatch run_three_model_eval.slurm

# Direct execution
python scripts/evaluate_three_models.py \
    --config configs/config_parl.yaml \
    --test_data data/test_data_100_enhanced.json \
    --output results/four_model_comparison.json
```

**Output**:
```json
{
  "summary": {
    "total_samples": 100,
    "rankings": {
      "avatar_1st": 48,
      "sft_1st": 36,
      "baseline_1st": 12,
      "lora_1st": 2
    }
  }
}
```

**Expected Runtime**: ~3 hours (4 models × 100 samples + GPT-4 judging)

---

#### `comprehensive_model_comparison.py`
**Purpose**: Complete evaluation framework for all comparison types

**What it does**:
- Runs all evaluation types (two-way, four-way)
- Supports multiple judge models
- Generates comparison reports
- Creates visualization plots

**Usage**:
```bash
python scripts/comprehensive_model_comparison.py \
    --config configs/config_parl.yaml \
    --test_data data/test_data_100_enhanced.json \
    --output_dir results/ \
    --comparison_types "two_way,four_way" \
    --judge_models "gpt-4.1-mini,gpt-4o-mini"
```

**Output**:
- Multiple result JSON files
- Comparison report (markdown)
- Visualization plots (PNG)

---

### Analysis & Utility Scripts

#### `view_perturbation_results.py`
**Purpose**: Interactive viewer for perturbation evaluation results

**Features**:
- Summary statistics by category (easy/medium/hard)
- Per-variant accuracy breakdown
- Difficulty gradient analysis
- Filter by correctness, category, variant type
- Sample-level inspection

**Usage**:
```bash
# View summary with comparison
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --compare

# View specific category
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --category negative_medium

# View incorrect samples only
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --incorrect-only \
    --limit 20

# Export to CSV
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --export results/analysis.csv
```

**Output Example**:
```
========================================
Perturbation Difficulty Evaluation Results
========================================

Overall Statistics:
  Total samples: 1,300
  Correct: 284 (21.85%)
  Incorrect: 1,016 (78.15%)

Accuracy by Category:
  Original:   23.00% (23/100)
  Positive:   N/A (0/0) ⚠️ Missing
  Easy:       23.00% (92/400)
  Medium:     20.25% (81/400)
  Hard:       22.00% (88/400)

Most Challenging Perturbations:
  negative_5 (Code hints):  16.00% (-7.00% vs original) 🔥
  negative_3 (Char repeat): 20.00% (-3.00% vs original)
  negative_11 (Curly):      20.00% (-3.00% vs original)
```

---

#### `view_eval_results.py`
**Purpose**: Interactive viewer for model comparison results

**Features**:
- Model comparison statistics
- Win/loss/tie breakdown
- Filter by model, correctness, sample ID
- Export to various formats
- Statistical analysis

**Usage**:
```bash
# View two-way comparison
python scripts/view_eval_results.py \
    results/sft_vs_baseline_evaluation.json

# Filter by specific model
python scripts/view_eval_results.py \
    results/four_model_comparison.json \
    --model avatar

# View winner distribution
python scripts/view_eval_results.py \
    results/four_model_comparison.json \
    --show-winners

# Export to CSV
python scripts/view_eval_results.py \
    results/sft_vs_baseline_evaluation.json \
    --export results/comparison.csv
```

---

#### `check_data_format.py`
**Purpose**: Validate data format and show statistics

**What it does**:
- Checks JSON format validity
- Validates required fields
- Counts variants and samples
- Shows dataset distribution
- Detects duplicates and missing data

**Usage**:
```bash
# Check training data
python scripts/check_data_format.py data/training_data_100.json

# Check with verbose output
python scripts/check_data_format.py \
    data/training_data_100.json \
    --verbose

# Export validation report
python scripts/check_data_format.py \
    data/training_data_100.json \
    --report data/validation_report.txt
```

**Output Example**:
```
✅ Valid JSON format
✅ All 100 samples present
✅ All samples have 20 variants (1 + 7 + 12)
✅ All ground_truth fields are non-empty
✅ All variant keys present: positive_1-7, negative_1-12
✅ No duplicate IDs

Dataset Distribution:
  ToolACE: 40 (40%)
  XLam: 40 (40%)
  Hammer: 20 (20%)

Domain Distribution:
  PDF: 30 (30%)
  API: 25 (25%)
  Database: 20 (20%)
  File: 15 (15%)
  Other: 10 (10%)
```

---

#### `validate_and_clean_data.py`
**Purpose**: Clean data by removing invalid samples

**What it does**:
- Removes samples with missing fields
- Removes samples with invalid ground_truth
- Removes duplicate IDs
- Validates variant counts
- Saves cleaned data

**Usage**:
```bash
# Clean training data
python scripts/validate_and_clean_data.py \
    --input data/training_data_100.json \
    --output data/training_data_100_clean.json

# Dry run (show what would be removed)
python scripts/validate_and_clean_data.py \
    --input data/training_data_100.json \
    --dry-run

# Strict mode (remove any sample with issues)
python scripts/validate_and_clean_data.py \
    --input data/training_data_100.json \
    --output data/training_data_100_clean.json \
    --strict
```

**Output**:
```
Processing: data/training_data_100.json
Found 100 samples

Validation Issues:
  ❌ Sample 'toolace_123': Missing negative_5 variant
  ❌ Sample 'xlam_456': Empty ground_truth
  ❌ Sample 'hammer_789': Duplicate ID

Removed: 3 samples
Remaining: 97 samples
Saved to: data/training_data_100_clean.json
```

---

#### `full_pipeline.py`
**Purpose**: End-to-end training and evaluation pipeline

**What it does**:
1. Data preprocessing and validation
2. Model training (SFT or AVATAR)
3. Model evaluation (multiple comparisons)
4. Result analysis and reporting

**Usage**:
```bash
# Run complete pipeline
python scripts/full_pipeline.py \
    --config configs/config_parl.yaml \
    --mode sft \
    --output_dir experiments/exp_001

# Run with AVATAR
python scripts/full_pipeline.py \
    --config configs/config_parl.yaml \
    --mode avatar \
    --output_dir experiments/exp_002

# Evaluation only (skip training)
python scripts/full_pipeline.py \
    --config configs/config_parl.yaml \
    --mode eval_only \
    --checkpoint checkpoints/sft_model \
    --output_dir experiments/exp_003
```

**Parameters**:
- `--config`: Configuration file
- `--mode`: Pipeline mode (sft/avatar/eval_only)
- `--output_dir`: Experiment output directory
- `--skip_training`: Skip training step
- `--skip_evaluation`: Skip evaluation step

---

## 🚀 Quick Start Examples

### Training a Model
```bash
# Train SFT model
sbatch run_sft_train.slurm

# Train AVATAR (instruction optimization)
sbatch run_baseline_avatar_fixed.slurm
```

### Running Evaluations
```bash
# Evaluate perturbation difficulty
sbatch run_baseline_perturbation_eval.slurm

# Compare SFT vs Baseline
sbatch run_sft_vs_baseline_eval.slurm

# Four-way comparison
sbatch run_three_model_eval.slurm
```

### Analyzing Results
```bash
# View perturbation results
python scripts/view_perturbation_results.py \
    results/baseline_perturbation_difficulty_eval.json \
    --compare

# View model comparison
python scripts/view_eval_results.py \
    results/sft_vs_baseline_evaluation.json
```

---

## 📊 Performance Benchmarks

| Script | Runtime | GPU Memory | Output Size |
|--------|---------|------------|-------------|
| `run_sft_training.py` | ~1 hour | 6GB | Checkpoint (~5.6GB) |
| `run_baseline_avatar.py` | ~30 min | 2GB | JSON (~1MB) |
| `evaluate_perturbation_difficulty.py` | ~7.5 hours | 6GB | JSON (~2.7MB) |
| `evaluate_sft_vs_baseline.py` | ~2 hours | 12GB | JSON (~1MB) |
| `evaluate_three_models.py` | ~3 hours | 20GB | JSON (~2MB) |

---

## 🔗 Related Files

- **Main README**: [../README.md](../README.md)
- **Data README**: [../data/README.md](../data/README.md)
- **Project Structure**: [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)
- **Documentation**: [../docs/](../docs/)

---

**Last Updated**: 2025-10-23
