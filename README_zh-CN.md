<div align="center">

# AgentRobustBench

[English](README.md) | **简体中文**

**A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents**

用于评测工具使用智能体在部署变化下的稳健性的基准，以及结合领域随机化的强化学习训练方法

NeurIPS 2026 · Evaluations & Datasets Track

[![arXiv 论文](https://img.shields.io/badge/arXiv-2605.11928-b31b1b)](https://arxiv.org/abs/2605.11928)
[![Hugging Face 数据集](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench)
[![在线排行榜](https://img.shields.io/badge/Hugging%20Face-Leaderboard-7C3AED)](https://huggingface.co/spaces/willchow66/AgentRobustBench)

**199 个原始任务 · 22 种扰动 · 3,721 个评测案例 · 已评测 21 个模型**

[运行六个模型评测样本](#运行六个模型评测样本) · [基准介绍](#基准介绍) · [实验结果](#实验结果) · [模型权重](#模型权重) · [完整评测](#完整评测) · [训练复现](#训练复现) · [引用](#引用)

</div>

AgentRobustBench 评测语言智能体在实际部署变化下能否正确使用工具，例如用户输入有错别字、工具名称容易混淆，或工具调用出现超时。我们同时提供 ToolRL-DR 训练方法，通过在训练中变化工具使用轨迹，提高模型的稳健性。

## 运行六个模型评测样本

这一步会实际运行模型，生成三个干净样本和三个错别字样本的预测，然后进行评分。下面的本地部署示例使用 **Linux、Python 3.11 或 3.12、Git，以及 vLLM 支持的 NVIDIA GPU**。请先查看 [vLLM 的硬件与驱动要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。首次安装依赖和下载模型需要额外时间。

终端就是输入命令并按回车执行的窗口。下面会使用**同一台机器上的两个终端**：终端 1 运行模型服务，终端 2 发送评测请求并评分。

### 第一步：下载项目并安装环境

在**终端 1** 中检查工具：

```bash
python3 --version
git --version
nvidia-smi
```

Python 应显示 3.11 或 3.12，`nvidia-smi` 应列出 GPU。如果找不到某条命令，请先安装相应工具，或使用已经配置好的 GPU 机器。

将仓库下载到用户主目录，建立独立的 Python 环境，并安装依赖：

```bash
git clone --depth 1 https://github.com/WillChow66/AgentRobustBench.git ~/AgentRobustBench
cd ~/AgentRobustBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements.txt
```

如果已经下载过项目，直接进入已有的项目目录，从 `python3 -m venv .venv` 开始。评测数据已随仓库提供，无需另外去 Hugging Face 下载。

### 第二步：启动模型服务

在**终端 1** 中运行：

```bash
vllm serve willchow66/ToolRL-DR-Full-Qwen2.5-3B --port 8000 --max-model-len 8192
```

vLLM 会自动从 Hugging Face 下载并加载模型，然后等待评测请求。后续运行通常会复用本地缓存。请保持这个终端运行。在服务启动完成后，打开**同一台机器上的终端 2**：

```bash
cd ~/AgentRobustBench
source .venv/bin/activate
python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://localhost:8000/v1/models'))['data'][0]['id'])"
```

最后一行应打印 `willchow66/ToolRL-DR-Full-Qwen2.5-3B`。如果出现连接错误，先检查终端 1，等待模型服务启动完成。

### 第三步：生成六个样本的预测

在**终端 2** 中整段粘贴：

```bash
python scripts/run_eval.py \
    --model-name willchow66/ToolRL-DR-Full-Qwen2.5-3B \
    --endpoint http://localhost:8000/v1 \
    --data-dir hf_data/datasets/api_eval \
    --files clean.jsonl realistic_typos.jsonl \
    --limit 3 --temperature 0 --max-workers 1 \
    --output-dir results/quick-demo
```

这会分别运行三个干净样本及其错别字版本。运行结束时，两个文件都应显示 `processed=3` 和 `errors=0`。结果保存到：

```text
results/quick-demo/
  clean.predictions.jsonl
  realistic_typos.predictions.jsonl
  _run_summary.json
```

每个预测文件中，一行对应一个样本，包含模型的输出。如果 `errors` 不为零，请先检查预测文件中的 `error` 字段和模型服务日志。

### 第四步：评分并查看结果

仍在**终端 2** 中运行：

```bash
python leaderboard/scoring/eval_results.py \
    --predictions results/quick-demo \
    --output results/quick-demo/scores.json \
    --per-sample-output results/quick-demo/per_sample_scores.jsonl
```

控制台应打印 `n=6`，表示已对六个样本评分，同时显示准确率。`scores.json` 保存汇总结果，`per_sample_scores.jsonl` 保存每个样本的评分。这六个样本用于跑通流程；与论文进行完整比较时，请继续运行下方的完整评测。评分使用确定性规则，无需另一个模型充当裁判。

如果想试 ToolRL-DR-Mixed，将两个终端中的模型 ID 都换成 `willchow66/ToolRL-DR-Mixed-Qwen2.5-3B`，并使用新的输出目录，例如 `results/quick-demo-mixed`。

### 如果已有模型服务：只安装评测客户端

如果模型服务已经在另一台机器上运行，Mac 或 Windows 电脑也可以发送评测请求。安装 Python 3.11 或 3.12 和 Git、下载仓库后，在项目目录中建立客户端环境。

Mac / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-client.txt
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-client.txt
```

Windows 中，将评测命令开头的 `python` 换成 `.\.venv\Scripts\python.exe`，并把每条命令写在同一行，去掉用于换行的行末 `\`。

随后执行第三、四步，将 `--endpoint` 改成你能访问的服务地址，以 `/v1` 结尾；将 `--model-name` 改成服务中注册的模型 ID。`localhost` 代表当前执行命令的电脑，只有服务与客户端在同一台机器时才能使用。如果服务器需要认证，通过 `--api-key` 传入对应密钥。本地默认 vLLM 无需 OpenAI 账号或密钥。

## 基准介绍

![AgentRobustBench 的四类扰动示意图](docs/assets/benchmark-overview.webp)

| 类别 | 改变的内容 | 示例 |
| --- | --- | --- |
| Observation：观测 | 智能体看到的输入 | 用户问题中出现错别字 |
| Action：动作 | 可选择的工具 | 加入同名的干扰工具 |
| Reward-relevant metadata：影响工具选择的元信息 | 引导工具选择的信息 | 误导性的工具名称或响应时间说明 |
| Transition：状态转移 | 工具调用后的返回结果 | 首次调用出现暂时性超时 |

[Hugging Face 数据集](https://huggingface.co/datasets/AgentRobustBench/AgentRobustBench) 包含 **2,718 条静态记录**。默认的单轮评测运行其中 **2,527 条**，跳过 **191 条多轮记录**，再增加 **1,194 个运行时案例**（199 个干净任务 × 6 种错误），合计 **3,721 个评测案例**。可以在[在线排行榜](https://huggingface.co/spaces/willchow66/AgentRobustBench)查看模型和各类扰动的得分。

## 实验结果

![不同模型规模下的准确率保持比例](docs/assets/robustness-retention.webp)

图中比较模型在 Observation、Action 和 Reward 扰动下相对干净样本保留了多少准确率。在 3B 参数规模下，ToolRL-DR-Full 的保持比例高于同规模 ToolRL 基线。在单独的运行时评测中，它在未使用 Transition 错误训练的情况下，将对应准确率差距缩小了约 27%。指标定义与实验设置见[论文](https://arxiv.org/abs/2605.11928)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `hf_data/datasets/api_eval/` | 17 个 JSONL 文件，包含干净样本和 16 种静态扰动；6 种 Transition 扰动由运行器在评测时生成。 |
| `scripts/` | 推理运行器、解析脚本、依赖和模型配置，以及 SLURM 作业模板。详见[评测指南](scripts/README.md)。 |
| `leaderboard/` | Gradio 排行榜应用、确定性评分器和公开分数表。 |
| `training/` | ToolRL-DR-Full 和 ToolRL-DR-Mixed 的训练数据、数据转换脚本、奖励函数补丁、训练启动脚本和超参数。详见[训练指南](training/README.md)。 |

## 模型权重

| 模型 | 训练数据 | 下载与模型说明 |
| --- | --- | --- |
| ToolRL-DR-Full | 扰动后的轨迹 | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Full-Qwen2.5-3B) |
| ToolRL-DR-Mixed | 干净轨迹与扰动轨迹混合 | [Hugging Face](https://huggingface.co/willchow66/ToolRL-DR-Mixed-Qwen2.5-3B) |

两个模型均基于 Qwen2.5-3B-Instruct，使用 [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)。

## 完整评测

先完成六个样本的流程，再按照[完整评测命令](scripts/README.md#full-benchmark-evaluation)运行所有静态文件和六种运行时扰动，最后对整个输出目录评分。每个模型、每次实验使用独立的输出目录。

集群用户可以使用 [`scripts/slurm/`](scripts/slurm/) 中的模板。提交作业前，需要替换为自己集群的账户、路径和调度器设置。

## 训练复现

训练用的 Parquet 文件已经放在 `training/data/` 中。按照[训练指南](training/README.md)完成 ToolRL 安装、奖励函数补丁和 `PYTHONPATH` 配置后，运行：

```bash
TRAIN_FILE=training/data/train.parquet \
VAL_FILE=training/data/test.parquet \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-full \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh
```

训练 Mixed 时，将 `TRAIN_FILE` 换成 `training/data/mixed_train.parquet`。训练需要额外的 GPU 资源和时间。

## 引用

下面保留当前仓库提供的 arXiv 引用条目：

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

## 许可证

代码使用 Apache-2.0。数据的来源许可证见论文和各数据来源的说明。
