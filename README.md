<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figs/axisrl-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/figs/axisrl-logo-light.svg">
    <img alt="AxisRL - Agentic Post-Training" src="docs/figs/axisrl-logo-light.svg" width="680">
  </picture>

  <p>
    <strong>English</strong> ·
    <a href="docs/README-cn.md">简体中文</a>
  </p>

  <p>
    <a href="https://github.com/XYZ-AI-Lab/axrl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/XYZ-AI-Lab/axrl/actions/workflows/ci.yml/badge.svg"></a>
    <img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-blue.svg">
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <a href="https://github.com/XYZ-AI-Lab/axrl/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  </p>

  <p>
    <a href="https://xyz-lab.ai">XYZ AI Lab</a> ·
    <a href="https://xyz-lab.ai/blogs/ai4ai-at-scale/">Technical Blog</a>
  </p>
</div>

# AxisRL

AxisRL is an agentic RL post-training framework built on SGLang rollout, Megatron training, and real-world agent workflows.

AxisRL connects high-throughput rollout, large-scale training, weight synchronization, data movement, resource scheduling, and reproducible debugging inside one coherent framework. SGLang and Megatron remain the core serving and training engines; AxisRL handles the system layer around agentic post-training.

## ✨ Highlights

- Built on **SGLang** for high-throughput rollout and **Megatron** for large-scale distributed training.
- Used in agent RL workflows with 300+ turn trajectories and training runs at hundreds-of-billions-parameter scale.
- Provides configurable policy optimization objectives including PPO, GRPO/GRPO2, GSPO, TOPR, TIS, and related variants.
- Supports both white-box agent environments and black-box harness capture through an OpenAI-compatible proxy.
- Reduces rollout/training idle time with partial rollout and a lightweight control plane.
- Provides handle-based data movement, context packing, routing replay, mismatch analysis, and spike replay for rollout-trainer consistency.

## 🧭 Why AxisRL?

LLM post-training workloads are moving beyond single-turn question answering. In agentic RL, a model may interact with a long-lived environment, call tools, observe tool results, update context, and receive a reward only after several turns.

That changes the job of a post-training framework. It has to coordinate multi-turn rollout, environment state, tool calls, verifiers, reward collection, training sample construction, and weight synchronization. It also has to make training behavior observable, because small differences in tokenization, chat templates, logprobs, routing, packing, or weight sync can appear later as loss spikes, reward instability, or rollout-trainer mismatch.

AxisRL is designed for this setting: real agent workflows, SGLang rollout, Megatron training, and the system contracts between them.

## 🏗️ Architecture

![AxisRL workflow](docs/figs/axrl-workflow.png)

At a high level, an AxisRL run follows this loop:

1. Rollout actors execute task-specific agent workflows.
2. SGLang workers serve model generation.
3. Environments, tools, verifiers, or external harnesses produce interaction records and rewards.
4. Megatron workers consume training samples and run PPO or GRPO-family training.
5. Updated weights are synchronized back to the rollout side for the next iteration.

AxisRL keeps the driver lightweight. The driver manages scheduling, lifecycle, metrics, phase transitions, and metadata. Heavy payloads, such as routing replay data or future multimodal artifacts, move through a handle-based data path and are read by trainer workers on demand.

## 🎯 Design Goals

| Goal | Problem | AxisRL Approach |
| --- | --- | --- |
| Flexibility | Agent workflows differ in control flow, tools, rewards, context management, and resource needs. | Use recipes for task logic, support white-box environments and black-box harness capture, and manage heterogeneous components through resource groups. |
| Efficiency | Long-tail trajectories, tool latency, verifiers, and repeated context can leave rollout or training resources idle. | Use partial rollout, thin control-plane scheduling, handle-based data movement, prefix-tree merge, MagiAttention, and off-policy stabilization tools such as TIS, sequence masking, and Icepop. |
| Observability | Rollout and trainer paths can silently diverge in tokenization, masks, logprobs, routing, packing, or weight versions. | Test critical boundaries and provide mismatch analysis, routing replay checks, and spike replay for reproducible debugging. |

## ⚙️ Installation

The recommended environment is the project Docker image. It includes SGLang, Megatron Core, MagiAttention, Ray, CUDA dependencies, and the Python packages used by the current recipes.

Pre-built image:

```bash
docker pull leejunjie/sglang-mcore:cu130-sgl0.5.14-mcore0.18-magi
```

Dockerfile:

```text
docker/cuda/cu130-sgl0.5.14-mcore0.18-magi.Dockerfile
```

Build locally if needed:

```bash
docker build \
  -f docker/cuda/cu130-sgl0.5.14-mcore0.18-magi.Dockerfile \
  -t axrl:cu130-sgl0.5.14-mcore0.18-magi \
  .
```

Start a container from the repository root:

```bash
docker run --gpus all --ipc=host --network=host --shm-size=64g -it \
  -v "$PWD":/workspace/axrl \
  -v "$HOME/axrl-data":/root/axrl-data \
  -w /workspace/axrl \
  leejunjie/sglang-mcore:cu130-sgl0.5.14-mcore0.18-magi \
  bash
```

Install AxisRL inside the container:

```bash
pip install -e .
```

Optionally download the common models and datasets referenced by the current recipes and tests:

```bash
python axrl/example/download_data.py
```

The bulk download can be large because it includes multi-billion-parameter models. For a narrower run, adjust the recipe model and dataset paths instead of downloading everything.

## 🚀 Quick Start

The recipe scripts below are the main entry points. They assume a GPU machine with enough resources for the default parallelism in each recipe. You can override most config fields from the command line with `--path.to.field=value`.

### GSM8K GRPO

```bash
AXRL_OUTPUT_DIR_NAME=grpo_gsm8k \
bash axis_recipe/grpo_gsm8k/run_train.sh \
  --online_rl_train.max_global_updates=4
```

### GSM8K PPO

```bash
AXRL_OUTPUT_DIR_NAME=ppo_gsm8k \
bash axis_recipe/ppo_gsm8k/run_train.sh \
  --online_rl_train.max_global_updates=4
```

### Search-R1

Search-R1 uses a retrieval server in addition to rollout and training workers.

```bash
export AXRL_SEARCH_PORT=18000
bash axis_recipe/search_r1/start_retriever.sh
python axis_recipe/search_r1/search_r1_config.py

AXRL_OUTPUT_DIR_NAME=search_r1 \
python -u axis_recipe/search_r1/train_search_r1.py \
  --config_path=axis_recipe/search_r1/search-r1-config.yaml \
  --online_rl_train.max_global_updates=4
```

For the default full recipe script:

```bash
AXRL_SEARCH_PORT=18000 \
bash axis_recipe/search_r1/run_train.sh
```

### Black-Box RL With OpenHands and E2B

This recipe is still a work in progress. It demonstrates the black-box harness integration path with OpenHands/E2B, but the config, launch scripts, and proxy interfaces may change.

The black-box RL recipe runs OpenHands inside E2B sandboxes. OpenHands calls AxisRL through an OpenAI-compatible proxy, and AxisRL captures model inputs, outputs, metadata, and rewards for training.

Prerequisites:

- `E2B_API_KEY` in the environment or in `.env`.
- `cloudflared` on the training host for the default tunnel path.
- An E2B template named `axrl-openhands`.

Build the E2B template once:

```bash
cd axis_recipe/blackbox_rl/e2b_template
e2b template build --name axrl-openhands
cd -
```

Run a small rollout smoke test:

```bash
AXRL_OUTPUT_DIR_NAME=blackbox-e2b-smoke \
AXRL_ROLLOUT_TEST_NUM_CASES=2 \
bash axis_recipe/blackbox_rl/run_rollout_test_distributed.sh
```

Run a short training job:

```bash
AXRL_OUTPUT_DIR_NAME=blackbox-e2b-train \
bash axis_recipe/blackbox_rl/run_train_distributed.sh \
  --online_rl_train.max_global_updates=4
```

More details are in [axis_recipe/blackbox_rl/README.md](axis_recipe/blackbox_rl/README.md).

## 🧩 Recipes

| Recipe | Mode | Entry Point | Notes |
| --- | --- | --- | --- |
| GSM8K GRPO | White-box RL | `axis_recipe/grpo_gsm8k/run_train.sh` | GRPO-style math training recipe. |
| GSM8K PPO | White-box RL | `axis_recipe/ppo_gsm8k/run_train.sh` | PPO math training recipe with actor and value workers. |
| Search-R1 | White-box tool RL | `axis_recipe/search_r1/run_train.sh` | Retrieval-augmented multi-turn search recipe. |
| Black-Box RL | Black-box harness RL | `axis_recipe/blackbox_rl/run_train_distributed.sh` | WIP OpenHands/E2B recipe; config and proxy interfaces may change. |

Each recipe owns task-specific logic such as the dataset, environment loop, verifier, reward computation, metrics, and training configuration. The shared AxisRL path handles rollout scheduling, sample construction, trainer input, weight sync, and debugging.

## 🔧 Key Technical Ideas

### White-Box and Black-Box Agent Workflows

AxisRL supports two integration patterns.

| Mode | Best Fit | AxisRL Handles | User Focus |
| --- | --- | --- | --- |
| White-box RL | Math, Search, simple tool environments | Agent loop control, rollout scheduling, training sample construction | Environment, tools, verifier, reward |
| Black-box RL | OpenHands, browser tasks, complex external harnesses | Model I/O and reward capture through an OpenAI-compatible proxy | Harness launch, adapters, verifier, reward collection |

White-box recipes express the environment loop inside AxisRL. Black-box recipes let an existing harness call the model through an OpenAI-compatible API, while AxisRL captures the interaction and builds trainable samples from it.

### Partial Rollout

Multi-turn agent trajectories often have long-tail latency. Some samples finish quickly, while others may require many tool calls or slow verifier responses. AxisRL supports partial rollout so completed or partially completed samples can be handed to the trainer earlier, reducing waits caused by slow trajectories.

### Thin Control Plane and Handle-Based Data Plane

Large rollout-side payloads should not have to pass through a centralized driver. AxisRL keeps the driver focused on scheduling and metadata, while heavy data moves through handles. Trainer workers read payloads on demand, which is useful for MoE routing replay data, complex rollout artifacts, and future multimodal intermediates.

### Context Management and MagiAttention

Agent contexts are not always simple linear sequences. A recipe may retain recent tool outputs, hide older tool results, or replace parts of context with placeholders. AxisRL uses prefix-tree merge and MagiAttention to reduce repeated attention compute while preserving the context each turn saw during rollout.

The key invariant is that context management and packing should not change training semantics. Whether a long trajectory is merged into one sample or split into multiple samples because of length limits, the training side should produce consistent gradients.

### Rollout-Trainer Consistency

Training instability often comes from subtle differences between rollout and trainer execution paths. AxisRL puts critical boundaries under test, including tokenization, chat templates, weight sync, checkpointing, routing replay, RolloutTrace packing, prefix-tree merge, MagiAttention forward, the OpenAI proxy, and mismatch analysis.

Rollout Routing Replay (R3) is one example. In MoE post-training, R3 reduces expert routing mismatch between rollout and trainer, making KL and loss more stable. Routing payloads use the handle-based data path so the driver does not become a heavy-data relay.

![R3 mismatch analysis](docs/figs/r3-mismatch.png)

### Mismatch Analysis and Spike Replay

Mismatch analysis compares token-level differences across rollout and trainer paths, backends, configurations, and routing replay settings. It helps identify whether a problem is broad drift, a small number of outliers, or something concentrated in a specific token range, sequence type, or context layout.

Spike replay targets occasional gradient or loss spikes. AxisRL can save a snapshot of weights, optimizer state, data, and relevant routing information before a spike update. The same update can then be replayed for inspection instead of waiting for the next non-deterministic failure.

## 📁 Repository Layout

```text
axrl/                 Core framework code
axis_recipe/          Public recipes and task-specific integrations
docs/                 Design notes, blog posts, and technical reports
docker/               Dockerfiles for reproducible training environments
scripts/              Runtime helper scripts
tests/                Unit, integration, and consistency tests
```

## 📚 Documentation

- [English blog](docs/axrl-blog-en.md)
- [Chinese blog](docs/axrl-blog-cn.md)
- [Black-box RL recipe](axis_recipe/blackbox_rl/README.md)
- [Config parsing notes](docs/config-parsing.md)

## 🗺️ Roadmap

- Add more real-world agent recipes and public case studies.
- Improve asynchronous execution between rollout and training.
- Extend support for multimodal rollout artifacts.
