# AxisRL: A Post-Training Framework for the Agentic Era

> An online RL loop for real-world agent workflows, built on SGLang rollout and Megatron training.

LLM post-training workloads are moving beyond single-turn question answering and into more complex agentic workflows. A model no longer just reads a prompt and returns a response. It may run inside a long-lived environment, observe state, call tools, read tool results, update context, and eventually receive a reward.

That changes what a post-training framework has to handle. The problem is no longer just "generate samples, then train on them." The framework also has to track multi-turn interaction, tool calls, environment state updates, and reward collection.

In traditional single-turn post-training, rollout is close to batch generation. In agentic RL, rollout becomes a multi-step interaction. Assistant responses, tool calls, tool results, verifiers, rewards, and environment state updates all become part of the same trajectory. Each step can change the context seen by later steps, and it can make sequence length, latency, and training cost much less predictable.

Environments and external harnesses also become more complex. Tasks such as Math and Search can often be controlled directly by the training framework. Coding agents such as OpenHands and Codex, browser-based tasks, desktop workspaces, multi-process tool systems, and third-party platforms often have their own control flow. Re-implementing the full harness inside the training framework is not always practical, and it can introduce behavior drift. For these systems, the training framework is better off capturing the interaction between the model and the harness, together with the final reward.

At the same time, the consistency and verifiability of training behavior become harder to maintain. Tokenization, chat templates, rollout and trainer execution paths, weight sync, MoE routing, context packing, and algorithm implementations can all introduce subtle differences. Many of these issues do not crash immediately. They show up as training curve drift, gradient spikes, or unstable rewards.

AxisRL starts from this setting. It builds on SGLang for rollout and Megatron for training, and provides the agentic RL layer around them: workflow orchestration, sample construction, weight synchronization, resource scheduling, data movement, and reproducible debugging.

## What Is AxisRL?

AxisRL is a training framework for agentic LLM post-training. It connects SGLang rollout, Megatron training, and real-world agent workflows into a compact online RL loop. The goal is to make multi-turn rollout, training, weight synchronization, resource scheduling, and debugging work within one coherent system boundary.

GitHub repository: [github.com/XYZ-AI-Lab/axrl](https://github.com/XYZ-AI-Lab/axrl).

AxisRL's core positioning is simple: a small core with a complete online RL loop.

- **SGLang** handles high-throughput rollout and model serving.
- **Megatron** handles large-scale model training, parallelism, checkpointing, and model execution.
- **AxisRL** handles agent workflow orchestration, weight sync, resource scheduling, data movement, training observability, and task-specific recipes.

This keeps AxisRL's scope narrow. It does not replace the serving engine or the training engine. Instead, it focuses on the contracts between rollout, training, data movement, and debugging.

## Overall Workflow

![AxisRL Workflow](/docs/figs/axrl-workflow.png)

From a user's perspective, the main AxisRL workflow has five steps:

1. Rollout actors run agent workflows according to a recipe.
2. SGLang workers serve model generation.
3. Environments, tools, verifiers, or external harnesses produce interaction records and rewards.
4. Megatron workers consume training samples and run PPO or GRPO-family training.
5. Updated weights are synchronized back to the rollout side for the next online RL iteration.

There are a few important boundaries.

First, rollout and training are decoupled through queues. The rollout side continuously produces trainable samples, while the training side consumes batches according to its own parallelism strategy.

Second, the model-serving boundary is token-in, token-out. Raw text, chat messages, and tool schemas are tokenized before they enter model workers. This makes sample packing, routing replay, and context merge easier to keep consistent across rollout and trainer paths.

Third, the driver mostly handles scheduling and metadata. It does not act as the transport layer for large tensors. Heavy rollout-side payloads, such as MoE routing information, complex rollout artifacts, or future multimodal intermediate data, move through a handle-based data path and are read by trainers on demand.

## Three Design Goals

AxisRL is organized around three design goals: Flexibility, Efficiency, and Observability.

| Goal | Problem | AxisRL Approach |
|---|---|---|
| Flexibility | Agent workflows differ widely in control flow, reward design, context management, and resource requirements. | Use recipes to hold task logic, support both white-box environments and black-box harness capture, and manage heterogeneous components through resource groups. |
| Efficiency | Multi-turn trajectories, tool calls, verifiers, and complex context create GPU bubbles, data movement overhead, and repeated compute. | Use partial rollout to reduce synchronization waits caused by long-tail trajectories; use TIS, sequence masking, and Icepop to improve off-policy training stability; use a thin driver and handle-based data path for heavy payloads so the driver does not become a single bottleneck; reduce repeated attention compute with prefix-tree merge and MagiAttention. |
| Observability | Rollout and trainer may compute different logprobs, masks, or routing decisions for the same tokens without crashing, causing loss spikes, unstable rewards, or regressions. | Turn key paths such as tokenization, chat templates, routing replay, and packing into tests, and provide mismatch analysis and spike replay to make issues easier to find, reproduce, and diagnose. |

AxisRL treats Flexibility and Efficiency as first-class goals, but it also makes training observability part of the system design. Critical paths such as tokenization, chat templates, weight sync, routing replay, and packing need tests, comparison tools, and reproducible debug inputs. This reduces the cost of finding and diagnosing bugs, and lowers regression risk during fast iteration.

## Agent Workflow Integration: White-Box and Black-Box

Agentic RL does not have a single integration pattern.

In white-box RL, AxisRL controls the agent loop. The user implements a gym-like environment: the model generates an action from the current observation, the environment executes the action, and then returns a new observation and reward. This works well when the environment is clear, the tool surface is limited, and the control flow can be expressed directly inside the training framework. Math, Search, and simple code execution environments fit this pattern.

In black-box RL, AxisRL does not require the user to re-implement the full control flow of an external harness. The harness can call the model through an OpenAI-compatible API. AxisRL's proxy captures model inputs, model outputs, and necessary metadata, then combines them with the final reward to construct training samples. This pattern is a better fit for OpenHands, browser tasks, multi-process tool systems, and existing benchmark harnesses.

| Mode | Best Fit | AxisRL Handles | User Focus |
|---|---|---|---|
| White-box RL | Math, Search, simple tool environments | Agent loop control, rollout scheduling, training sample construction | Environment, tools, verifier, reward |
| Black-box RL | OpenHands, browser tasks, complex external harnesses | Model I/O and reward capture through an OpenAI-compatible proxy | Harness launch, adapters, verifier and reward collection |

This distinction matters for integration cost and behavior consistency. Many real agent systems derive much of their value from complex harnesses. If a training framework requires those harnesses to be fully rewritten, integration becomes expensive and behavior can drift. With both paths, simple environments can be integrated directly, while complex systems can enter the RL loop through black-box capture.

## Recipes, Run Modes, and Resource Scheduling

AxisRL's user entry point is a set of recipes, not a single global harness.

Each recipe can define the dataset, rollout loop, environment, verifier, reward computation, metrics, and training configuration for a task. For white-box RL, a recipe usually implements the environment interaction logic. For black-box RL, it focuses more on launching the external harness, capturing model interactions, and collecting rewards.

This structure leaves room for task-specific control flow. Different tasks can use different environment logic, as long as they eventually produce a common training sample format that can reuse the same training path.

AxisRL supports several run modes:

- rollout-only: run rollout and evaluation only, useful for checking environments, rewards, and throughput.
- train-only: start training from existing samples or offline data.
- rollout + train: run the full online RL loop.
- eval-only: evaluate model behavior only.
- mismatch analysis: fix data and configuration, then compare rollout and trainer behavior end to end.

For resource scheduling, SGLang workers, Megatron workers, verifiers, reward models, teacher models, and black-box adapters can be managed through Ray actors and resource groups. Rollout, training, verification, and external services do not have to be tied to the same resource type. The system can place components according to the workload.

## Efficiency at Scale: Reducing GPU Bubbles in Multi-Turn Agents

One core efficiency problem in multi-turn agent training is the GPU bubble.

Even in single-turn generation, length variance inside a batch can affect throughput. In agentic workflows, the variance is larger. One trajectory may finish quickly. Another may require dozens or hundreds of tool calls. Some verifiers return immediately, while some external harnesses have long-tail latency. If the system uses a strong synchronization boundary, both training and rollout can end up waiting for the slowest samples.

In this release, AxisRL supports partial rollout to reduce this waiting. The system does not have to wait for every trajectory to finish before moving data to the training side. Instead, completed or partially completed samples can be handed to the trainer earlier, so the trainer can consume available data sooner and wait less on long-tail trajectories.

AxisRL can also work with TIS, sequence masking, and Icepop to perform importance correction and filtering at the token and sequence levels, improving training stability under off-policy settings.

AxisRL has already run stably in agent RL workflows with more than 300 turns, as well as training runs for models at the hundreds-of-billions parameter scale.

## Thin Control Plane, Handle-Based Data Plane

Another scaling problem is data movement.

Many RL systems run into a hidden bottleneck as they scale: the driver process becomes a large-data relay. Rollout-side data flows through the driver and is then distributed to multiple trainer ranks. For ordinary metadata, this is usually fine. For large rollout-side payloads such as MoE routing information, multimodal artifacts, or complex context-related intermediate data, this can quickly become a network, serialization, and CPU bottleneck.

AxisRL keeps the control plane thin.

The driver handles scheduling, lifecycle management, metrics, phase transitions, and sample metadata. Large payloads go through a handle-based data plane. The rollout side places heavy data in local storage, an object store, or a tensor store, while the training sample only carries a handle and necessary metadata. A Megatron worker reads the payload on demand for its own batch, and the system tries to overlap communication with training compute where possible.

R3 is a representative use case for this data plane.

In MoE post-training, Rollout Routing Replay (R3) reduces expert routing mismatch between rollout and trainer, making KL and loss more stable. But routing information itself can be large. If it follows every sample through a centralized driver, the system cost can be high. AxisRL puts routing payloads on the handle-based data path, making R3 part of rollout-trainer end-to-end consistency while preventing the driver from becoming a heavy-data movement bottleneck.

R3 is only one example payload. Future multimodal inputs, intermediate features, video artifacts, or other large rollout-side data that the trainer needs can reuse the same mechanism.

## Context Management and MagiAttention

The context of an agentic workflow is often not a simple linear sequence.

A search agent may execute many tool calls. To control context length, the system may keep only the most recent tool results, while compressing, hiding, or replacing older tool results with placeholders. As a result, the context visible to the assistant at turn N is not necessarily a strict prefix of the context visible at turn N+1. The full trajectory looks more like a tree with shared prefixes.

If every turn is expanded into an independent training sample, the samples contain many repeated tokens, and attention computation is repeated as well. AxisRL uses prefix-tree merge to find shared token prefixes across turns, and uses MagiAttention to express tree-like attention. Shared parts can be computed with less duplication, while each turn still sees only the context it was allowed to see during rollout.

This serves both Flexibility and Efficiency. Flexibility comes from allowing recipes to define task-specific context retention, compression, and replacement policies. Efficiency comes from letting the trainer consume these contexts through prefix-tree merge and complex attention masks, without recomputing the same prefix for every turn.

The key invariant is that context management and packing should not change training semantics. Whether a long trajectory is merged into one sample or split into multiple samples because of length limits, the training side should produce consistent gradients.

## Observability: Correctness First, Mismatch Analysis, and Spike Replay

Agentic RL systems can fail in subtle ways, and many failures do not show up as immediate crashes. More often, they appear as abnormal training curves, KL spikes, unstable rewards, growing rollout-trainer mismatch, or gradient spikes.

AxisRL treats correctness as an engineering practice: tests, reproducible inputs, and analysis tools around the paths most likely to diverge.

### Correctness First: Tests Are Part of the Architecture

AxisRL puts important training behavior under test, including tokenization, chat templates, weight sync, checkpointing, routing replay, RolloutTrace packing, prefix-tree merge, MagiAttention forward, the OpenAI proxy, and mismatch analysis.

These tests are not just pre-release checks. They define critical boundaries between rollout, training, context management, and debug tooling. For a training system that is still evolving quickly, this lowers regression risk. It also gives coding agents clearer safety boundaries when they participate in implementation, refactoring, and experimentation.

### Mismatch Analysis and Spike Replay

Mismatch means that the rollout side and the training side produce different outputs for the same input tokens because of implementation differences. A common example is a logprob difference. These differences are often one source of training instability.

AxisRL provides mismatch analysis tools to compare token-level differences across backends, configurations, and routing replay settings. The goal is to help users quickly tell whether a problem is global drift, a small number of outliers, or something concentrated in a specific sequence type, token range, or context layout. The figure below shows a mismatch report for comparing rollout-trainer end-to-end consistency before and after enabling R3.

![R3 Mismatch Analysis](/docs/figs/r3-mismatch.png)

Spike replay targets a different failure mode: occasional gradient spikes or loss spikes during training. AxisRL can save a snapshot of weights, optimizer state, data, and relevant routing information before a spike update. Users can later reload the same snapshot, reproduce the spike on the same data, and inspect which samples, tokens, parameters, or routing patterns contributed to it.

This turns debugging from "wait for the next random spike" into "run repeatable experiments on the same spike." At large post-training scale, this reproducibility is often more useful than observing an anomaly once.

## Roadmap

AxisRL is still moving quickly. Next, we plan to focus on several areas:

| Area | Next Step |
|---|---|
| Agent recipes | Add more real-world agent workflows and public case studies. |
| Fully async controller | Decouple the execution pace of rollout and training, reducing synchronization waits and GPU idle time to improve overall training throughput. |
| Multimodal | Extend the training-side consumption path for large rollout-side artifacts. |

The long-term goal is to keep the core small and understandable while covering the system problems that are hardest, and most likely to go wrong, in agentic RL post-training: multi-turn rollout efficiency, rollout-trainer end-to-end consistency, flexible context management, large-scale training, and reproducible debugging.
