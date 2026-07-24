# AxisRL: Agentic 时代的后训练框架

> 在 SGLang rollout 和 Megatron training 之上，构建一个面向真实 agent workflow 的 online RL 训练闭环。

LLM 后训练的工作负载正在从单轮问答，逐步扩展到更复杂的 agentic workflow。模型不再只是读入一个 prompt，然后输出一段 response；它会在一个长时间运行的环境中反复观察状态、调用工具、读取结果、更新上下文，并最终得到一个 reward。

这让后训练系统的边界开始发生变化。

在传统单轮后训练里，rollout 更像 batch generation。到了 agentic RL，rollout 变成了一段可能持续很多轮的交互过程：assistant response、tool call、tool result、verifier、reward、环境状态更新都会进入同一个 trajectory。每一步都可能改变之后的上下文，也可能让样本长度、延迟和训练成本变得更不稳定。

环境和外部 harness 也变得更复杂。Math、Search 这类任务可以由训练框架直接控制 agent loop；但 OpenHands、Codex 这类 coding agent，或者浏览器、桌面工作区、多进程工具系统和第三方平台，往往有自己的 control flow。对这类系统，完整复刻 harness 不一定现实，也容易引入行为偏差；训练框架更适合捕获模型和 harness 之间的交互，以及最终 reward。

与此同时，训练行为的一致性和可验证性也更难维护。tokenization、chat template、rollout / trainer 执行路径、weight sync、MoE routing、context packing、算法实现，任何一个细节出错，都可能让端到端一致性出现难以解释的异常。很多错误不会立刻 crash，而是以训练曲线漂移、gradient spike 或 reward 不稳定的方式出现。

AxisRL 的出发点就是在这个背景下重新思考：如果站在 SGLang 和 Megatron 的肩膀上，是否可以构建一个尽可能简单、但足够高效、可测试、可扩展的 agentic RL 后训练框架？

## AxisRL 是什么

AxisRL 是一个面向 agentic LLM post-training 的训练框架。它连接 SGLang rollout、Megatron training 和真实 agent workflow，提供一个小核心但完整的 online RL 闭环，让多轮 rollout、训练、权重同步、资源调度和 debug analysis 可以在同一套系统边界内协同工作。

代码仓库：[github.com/XYZ-AI-Lab/axrl](https://github.com/XYZ-AI-Lab/axrl)。

它的核心定位可以概括为：小核心 + 完整 online RL 闭环。

- **SGLang** 负责高吞吐 rollout 和模型服务。
- **Megatron** 负责大模型训练、并行、checkpoint 和模型执行。
- **AxisRL** 负责 agent workflow、weight sync、resource scheduling、data movement、training observability，以及面向具体任务的 recipes。

这种定位让 AxisRL 聚焦在 agentic post-training 的系统连接层：边界更窄，数据契约更明确，也更容易验证。

## 整体 Workflow

![AxisRL Workflow](/docs/figs/axrl-workflow.png)

从用户视角看，AxisRL 的主流程可以概括为五步：

1. Rollout actors 根据 recipe 运行 agent workflow。
2. SGLang workers 负责模型生成。
3. Environment、tool、verifier 或外部 harness 产生交互记录和 reward。
4. Megatron workers 消费训练样本，执行 PPO / GRPO-family 训练。
5. 权重同步回 rollout 侧，进入下一轮 online RL。

这里有几个关键边界。

首先，rollout 和 training 通过 queue 解耦。rollout 侧只需要持续产生可训练样本，training 侧按自己的并行策略消费 batch。

其次，模型服务边界保持为 token-in-token-out：raw text、chat messages 和 tool schema 会在进入 model worker 前完成 tokenization。这样 sample packing、routing replay 和 context merge 更容易保持 rollout / trainer 端到端一致性。

最后，driver 主要处理调度和 metadata，不承担大型 tensor 的中转。像 MoE routing information、复杂 rollout artifact 或未来的多模态中间数据，会通过 handle-based data path 由 trainer 按需读取。

## 三个设计目标

AxisRL 的设计目标可以收敛成三件事：Flexibility、Efficiency 和 Observability。

| 目标 | 解决的问题 | AxisRL 的思路 |
|---|---|---|
| Flexibility | 不同 agent workflow 的 control flow、reward、context management 和资源需求差异很大 | 用 recipes 承载任务逻辑，支持 white-box env 和 black-box harness capture，并通过 resource groups 管理异构组件 |
| Efficiency | 多轮 trajectory、工具调用、verifier 和复杂上下文会带来 GPU bubble、数据搬运和重复计算 | 用 partial rollout 减少长尾 trajectory 带来的同步等待，并用 TIS、sequence masking、Icepop 等手段提高 off-policy 训练稳定性；同时用轻量级 driver 和 handle-based data path 处理重数据，避免 driver 成为单点瓶颈；用 prefix-tree merge / MagiAttention 降低复杂 attention 成本 |
| Observability | rollout / trainer 对同一批 tokens 的 logprob、mask 或 routing 计算不一致时，往往不会直接 crash，但会表现为 loss spike、reward 不稳定或 regression | 把 tokenization、chat template、routing replay、packing 等关键路径沉淀成 tests，并提供 mismatch analysis 和 spike replay，让问题更容易发现、复现和定位 |

AxisRL 关注 Flexibility 和 Efficiency，同样关注为 tokenization、chat template、weight sync、routing replay、packing 等关键路径提供测试、对比工具和可复现的 debug 输入，尽量降低 bug 的发现、复现和定位成本，并减少快速迭代中出现 regression 的风险。

## Agent Workflow 接入：White-box 与 Black-box

Agentic RL 不只有一种接入方式。

在 white-box RL 中，AxisRL 控制 agent loop。用户实现一个 gym-like environment：模型基于当前 observation 生成 action；环境执行 action，并返回新的 observation 和 reward。这种模式适合环境相对清晰、tool 数量有限、control flow 可以被训练框架直接表达的场景，例如 Math、Search、简单代码执行环境等。

在 black-box RL 中，AxisRL 不要求复刻外部 harness 的完整 control flow。外部 harness 可以通过 OpenAI-compatible API 请求模型，AxisRL 的 proxy 捕获模型输入、输出和必要 metadata，再结合最终 reward 构造训练样本。这种模式更适合 OpenHands、浏览器、多进程工具系统或已有 benchmark harness。

| 模式 | 适合场景 | AxisRL 负责 | 用户关注 |
|---|---|---|---|
| White-box RL | Math、Search、简单工具环境 | 控制 agent loop、调度 rollout、组织训练样本 | environment、tool、verifier、reward |
| Black-box RL | OpenHands、浏览器、复杂外部 harness | 通过 OpenAI-compatible proxy 捕获模型 I/O 和 reward | harness 启动、adapter、verifier / reward 收集 |

这个区分直接影响接入成本和行为一致性。很多真实 agent 系统的价值恰恰来自复杂 harness。如果训练框架必须完全重写这些 harness，接入成本会很高，行为也难以保持一致。通过这两条路径，简单环境可以直接 white-box 化，复杂系统也能通过 black-box capture 进入 RL 训练闭环。

## Recipes、运行模式与资源调度

AxisRL 的用户入口是一组 recipes，而不是单一的全局 harness。

每个 recipe 可以根据任务需要定义 dataset、rollout loop、environment、verifier、reward 计算、metrics 和训练配置。对 white-box RL，recipe 通常会实现环境交互逻辑；对 black-box RL，recipe 更关注如何启动外部 harness、如何捕获模型交互，以及如何收集 reward。

这种组织方式保留了足够的自由度。不同任务可以有完全不同的环境交互逻辑，只要最后能产出统一的训练样本，就可以复用同一套训练路径。

AxisRL 也支持不同运行模式：

- rollout-only：只运行 rollout 和评估，用于检查环境、reward 和吞吐。
- train-only：从已有 samples 或离线数据启动训练。
- rollout + train：完整 online RL 闭环。
- eval-only：只评估模型行为。
- mismatch analysis：固定数据和配置，分析 rollout / trainer 端到端一致性差异。

在资源调度上，SGLang workers、Megatron workers、verifier、reward model、teacher model 和 black-box adaptor 都可以通过 Ray actors 和 resource groups 管理。这样 rollout、training、verification 和外部服务不必绑定在同一类资源上，系统可以根据 workload 调整资源布局。

## Efficiency at Scale：减少多轮 Agent 的 GPU Bubble

多轮 agent 训练里的一个核心效率问题是 GPU bubble。

单轮 generation 中，batch 内样本的长度差异已经会影响吞吐；到了 agentic workflow，差异会进一步放大。一个 trajectory 可能很快结束，另一个 trajectory 可能需要几十甚至上百轮工具调用；某些 verifier 很快返回，某些外部 harness 可能有很长尾的响应时间。如果系统采用过强的同步边界，训练和 rollout 很容易被最慢的一批样本拖住。

这次 release 中，AxisRL 支持 partial rollout 来缓解这个问题。系统不必等待所有 trajectory 都完整结束后才向训练侧推进，而是可以把已经完成或阶段性完成的样本更早交给 trainer，让训练侧更早消费可用数据，减少长尾 trajectory 造成的等待。

同时，AxisRL 还可以配合 TIS、sequence masking、Icepop 等手段，在 token 和 sequence 级别做重要性校正和筛选，缓解 off-policy 带来的训练不稳定。

目前 AxisRL 已经稳定用于超过 300 turns 的 agent RL workflow，以及数百 B 参数级模型的训练场景。

## Thin Control Plane, Handle-Based Data Plane

另一个规模化问题是数据搬运。

很多 RL 系统在规模变大后会遇到一个隐性瓶颈：driver process 被迫成为大数据中转站。rollout 侧产生的数据先经过 driver，再分发到多个 trainer rank。对于普通 metadata，这个设计问题不大；但对于大型 rollout-side payload，例如 MoE routing information、多模态 artifact 或复杂上下文相关中间数据，这会迅速变成网络、序列化和 CPU bottleneck。

AxisRL 的设计是让 control plane 尽可能薄。

driver 负责调度、生命周期管理、metrics、阶段切换和 sample metadata。大型 payload 则进入 handle-based data plane：rollout 侧把重数据放在本地 storage、object store 或 tensor store 中，训练样本里只保留 handle 和必要 metadata。Megatron worker 在真正需要时，根据自己的 batch 按需读取，并尽量让通信和训练计算 overlap。

R3 是这条 data plane 的一个代表性用例。

在 MoE post-training 中，Rollout Routing Replay (R3) 用来减少 rollout / trainer 之间的 expert routing mismatch，让 KL 和 loss 更稳定。但 routing information 本身可能很大，如果它跟随 sample 经过中心化 driver，系统成本会很高。AxisRL 把 routing payload 放进 handle-based data path，让 R3 成为 rollout / trainer 端到端一致性的一部分，同时避免把 driver 变成重数据搬运瓶颈。

R3 只是这条路径的典型 payload。未来的多模态输入、中间特征、视频 artifact 或其他 trainer 侧需要消费的大型 rollout-side 数据，也可以复用同一套机制。

## Context Management 与 MagiAttention

Agentic workflow 的上下文往往不是一条简单线性序列。

一个搜索 agent 可能执行很多次 tool call。为了控制上下文长度，我们可能只保留最近几次 tool results，把更早的工具结果压缩、隐藏或替换成 placeholder。这样一来，第 N 轮 assistant 看到的上下文，不一定是第 N+1 轮上下文的严格前缀。整个 trajectory 更像一棵共享前缀的树。

如果直接把每个 turn 展开成独立 training sample，训练样本会包含大量重复 token，attention 计算也会被重复执行。AxisRL 使用 prefix-tree merge 找到不同 turns 之间共享的 token 前缀，并用 MagiAttention 表达 tree-like attention：共享部分尽量只计算一次，但每个 turn 仍然只能看到它在 rollout 时应该看到的上下文。

这同时服务于 Flexibility 和 Efficiency。Flexibility 来自 recipe 可以按任务需要定义上下文保留、压缩和替换策略；Efficiency 来自训练侧可以用 prefix-tree merge 和复杂 attention mask 消费这些上下文，而不必为每一轮重复计算相同前缀。

这里的关键不变性是：context management 和 packing 不应该改变训练语义。无论一个长 trajectory 被 merge 成一个 sample，还是因为长度限制被切成多个 samples，训练侧都应该得到一致的gradient。

## Observability：Correctness First、Mismatch Analysis 与 Spike Replay

Agentic RL 系统很容易在细节上出错，而且很多错误不会立刻表现为 crash。它们更常见的表现是训练曲线异常、KL spike、reward 不稳定、rollout / trainer mismatch 变大或 gradient spike。

AxisRL 把 correctness 具体化为可测试、可复现、可分析的工程路径。

### Correctness First：测试是架构的一部分

AxisRL 把重要训练语义纳入测试覆盖，包括 tokenization、chat template、weight sync、checkpoint、routing replay、RolloutTrace packing、prefix-tree merge、MagiAttention forward、OpenAI proxy、mismatch analysis 等。

这些测试不仅是发布前的检查项，也是系统架构的一部分。它们定义了 rollout、training、context management 和 debug tooling 之间的关键边界。对于一个仍在快速迭代的训练系统来说，这能降低 regression 风险；对于 AI4AI development 来说，这也让 coding agents 参与实现、重构和实验时有更明确的安全边界。

### Mismatch Analysis 与 Spike Replay

Mismatch 指 rollout 侧和 training 侧因为实现细节不同，对同一批 input tokens 产生不同输出，例如 logprob 差异。这类差异往往是训练不稳定的重要原因之一。

AxisRL 提供 mismatch analysis 工具，用来比较不同 backend、不同配置和不同 routing replay 设置下的 token-level 差异。目标是帮助用户快速判断：问题是整体漂移、少数 outlier，还是集中在某类 sequence、token 或 context layout 上。下图是一个 mismatch report，用来观察启用 R3 前后 rollout / trainer 端到端一致性的变化。

![R3 Mismatch Analysis](/docs/figs/r3-mismatch.png)

Spike replay 则服务于另一类问题：训练中偶发的 gradient spike 或 loss spike。AxisRL 支持在 spike 更新前保存 weight、optimizer、data 和相关 routing information 的 snapshot。之后可以加载同一个 snapshot，在相同数据上复现 spike，并分析它来自哪些 samples、tokens、parameters 或 routing patterns。

这让 debug 从“等待下一次随机 spike”变成“对同一个 spike 做可重复实验”。在大规模后训练里，这种可复现性往往比一次性观察到异常更重要。

## Roadmap

AxisRL 仍在快速迭代中。接下来我们会继续围绕几个方向推进：

| 方向 | 下一步 |
|---|---|
| Agent recipes | 扩展更多真实 agent workflow 和公开 case study |
| Fully async controller | 解耦 rollout 和 training 的执行节奏，减少同步等待和 GPU 空转，提高整体训练吞吐 |
| Multimodal | 扩展大型 rollout-side artifact 的训练侧消费路径 |

长期目标是保持核心足够小、足够可理解，同时持续覆盖 agentic RL 后训练里最难、也最容易出问题的系统环节：多轮 rollout 的效率、rollout / trainer 端到端一致性、灵活 context management、大规模训练和可复现 debug。
