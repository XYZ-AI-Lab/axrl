# Blackbox RL With OpenHands In E2B

This recipe is still a work in progress. It demonstrates the black-box harness integration path with OpenHands/E2B, but the config, launch scripts, and proxy interfaces may change.

This recipe trains AXRL with coding-agent rollouts. Each rollout starts
OpenHands inside an E2B sandbox, lets it solve a LeetCode-style Python task, and
routes its model calls back to the local AXRL rollout worker through an
OpenAI-compatible proxy.

## Runtime Path

1. AXRL starts an `OpenAIProxyServer` on the training host.
2. The default config starts a Cloudflare Quick Tunnel:

   ```bash
   cloudflared tunnel --url http://127.0.0.1:<proxy-port>
   ```

3. OpenHands runs in E2B with `LLM_BASE_URL` set to:

   ```text
   <tunnel-url>/sessions/<session_id>/v1
   ```

4. The proxy requires a per-run bearer token. The recipe generates it and passes
   it to OpenHands as `LLM_API_KEY`.
5. E2B outbound network access is restricted to the active proxy host. The
   sandbox cannot access the public internet by default.
6. AXRL converts OpenAI-compatible chat requests into rollout-worker requests,
   generates model responses locally, and returns them through the proxy.
7. The final solution file is read back from the E2B sandbox and scored by the
   recipe-local LeetCode verifier.

## Prerequisites

- An `E2B_API_KEY` in the environment or in `.env`.
- The Python dependencies installed from this repo, including the E2B SDK:

  ```bash
  pip install -e .
  ```

- `cloudflared` on the training host for the default tunnel path.
- The model and dataset files available under the normal AXRL data/model paths.
- An E2B template named `axrl-openhands`.

Build the E2B template once:

```bash
cd axis_recipe/blackbox_rl/e2b_template
e2b template build --name axrl-openhands
```

The template should contain OpenHands and the Python runtime needed to create
and run the generated solution file. It does not need the AXRL repo or model
weights.

## Smoke Eval

Run a small E2B smoke before starting training:

```bash
AXRL_OUTPUT_DIR_NAME="blackbox-e2b-smoke-$(date +%Y%m%d-%H%M%S)" \
AXRL_ROLLOUT_TEST_NUM_CASES=16 \
AXRL_ROLLOUT_WORKERS_TOTAL=2 \
AXRL_MEGATRON_DP_SIZE=1 \
bash axis_recipe/blackbox_rl/run_rollout_test_distributed.sh
```

This starts Ray, the local OpenAI-compatible proxy, the Cloudflare tunnel, the
SGLang rollout workers, and real E2B OpenHands sandboxes. The smoke command
writes a log to the launcher output directory:

```text
$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/run-rollout-test-blackbox-rl.log
```

It also writes per-case HTML reports under:

```text
$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/openhands_cases/
```

If `AXRL_OUTPUT_DIR` is not set before launching, the recipe uses
`${AXRL_SHM_ROOT:-$HOME/axrl-data}/outputs`.

Use the smoke output to confirm that:

- the Cloudflare tunnel exposes a `https://*.trycloudflare.com` URL;
- E2B sandboxes can reach only that tunnel host;
- OpenHands sends model requests through the proxy;
- the verifier returns scores without timeouts.

## Training

Start the default single-node training run:

```bash
AXRL_OUTPUT_DIR_NAME="blackbox-e2b-train-$(date +%Y%m%d-%H%M%S)" \
bash axis_recipe/blackbox_rl/run_train_distributed.sh \
  --online_rl_train.max_global_updates=132
```

The launcher writes:

```text
$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/run-train-blackbox-rl.log
$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/logs/
```

If `AXRL_OUTPUT_DIR` is not set before launching, the recipe uses
`${AXRL_SHM_ROOT:-$HOME/axrl-data}/outputs`.

TensorBoard logs are under:

```text
$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/logs/BlackBoxRL/$AXRL_OUTPUT_DIR_NAME
```

Open TensorBoard with:

```bash
tensorboard --logdir "$AXRL_OUTPUT_DIR/$AXRL_OUTPUT_DIR_NAME/logs"
```

Checkpoint saving is disabled in the default blackbox config to avoid filling
shared-memory-backed output directories. Set
`online_rl_train.checkpoint_every_n_global_updates` to a positive interval only
when the output directory has enough persistent storage.

## Default Scale

The default config is tuned for an 8-GPU single-node host:

- rollout workers: `2` workers, tensor parallel size `4`;
- actor/Megatron worker: `dp_size=1`, `tp_size=4`, `cp_size=2`;
- rollout actors: `32`;
- CPUs per rollout actor: `1`;
- maximum concurrent rollout requests: `32`;
- OpenAI adapter processors per rollout actor: `1`;
- verifier processors per rollout actor: `1`.

`32` is the validated stable default for the current Cloudflare Quick Tunnel
and E2B/OpenHands recipe path. It keeps one E2B-backed OpenHands rollout per
Ray actor and stays well below the practical `200` in-flight request limit of
Cloudflare Quick Tunnel free usage. If you use a managed tunnel or your own
public endpoint with a higher limit, you can raise
`controller.max_running_requests` and tune `controller.num_rollout_actors`
accordingly.
For OpenHands/E2B, keep `controller.num_rollout_actors` greater than or equal
to `controller.max_running_requests` unless you have validated multiple
sandbox-backed rollouts inside one Ray actor.

For smaller tests, reduce concurrency:

```bash
bash axis_recipe/blackbox_rl/run_train_distributed.sh \
  --controller.max_running_requests=32 \
  --controller.num_rollout_actors=32 \
  --controller.num_cpus_per_actor=4 \
  --openai_proxy.adapter_num_processors=1 \
  --verifier_num_processors=1 \
  --online_rl_train.max_global_updates=4
```

## Network Exposure

The default tunnel config is:

```yaml
openai_proxy:
  exposure:
    tunnel:
      command:
      - cloudflared
      - tunnel
      - --url
      - http://127.0.0.1:{port}
      ready_url_regex: https://[-a-zA-Z0-9.]+\.trycloudflare\.com
```

To use an operator-managed endpoint instead:

```yaml
openai_proxy:
  exposure:
    exposed_base_url: https://your-proxy.example.com
    tunnel: null
```

The recipe derives the E2B network allowlist from the final exposed URL host.
Broad allowlist entries such as `0.0.0.0/0`, `::/0`, `*`, or `all` are rejected.

## Useful Overrides

- Change the model:

  ```bash
  --megatron_worker.model.name=<hf-model> \
  --rollout_worker.model.name=<hf-model>
  ```

- Limit OpenHands calls per task:

  ```bash
  --openhands_env.max_model_calls=8
  ```

- Change the first OpenHands model-request startup window:

  ```bash
  --openhands_env.initial_request_timeout_seconds=360
  ```

- Limit generation length:

  ```bash
  --rollout_worker.sampling_config.max_new_tokens=4096
  ```

- Disable the initial full eval when you only want to test training mechanics:

  ```bash
  --online_rl_train.eval_on_start=false
  ```

## Troubleshooting

- `OpenAI proxy has no E2B-routable exposure`:
  configure `openai_proxy.exposure.tunnel` or set a reachable
  `openai_proxy.exposure.exposed_base_url`.

- `E2BRunner requires a non-empty network allow_out list`:
  the recipe could not derive a safe outbound allowlist. Check the proxy
  exposure config.

- `cloudflared` executable not found:
  install `cloudflared`, or set `openai_proxy.exposure.exposed_base_url` and
  `tunnel: null`.

- Many OpenHands tasks exit with code `-1`:
  inspect the generated HTML case reports. Common causes are invalid tool calls,
  hitting `openhands_env.max_model_calls`, or no model request arriving before
  the OpenHands request timeout.

- E2B sandboxes start but rollouts do not complete:
  the first model request from OpenHands may be slower than
  `openhands_env.initial_request_timeout_seconds`. The default is `360`
  seconds. If a runtime does not send its first request before that timeout,
  AXRL cleans it up and retries the rollout until a valid first model request
  arrives.

- Native thread creation errors from BLAS/tokenizers:
  use the provided launchers. They source
  `axis_recipe/blackbox_rl/launcher_env.sh`, which caps common native thread
  pools before Ray workers start.

- E2B SDK HTTP/2 `ConnectionState.CLOSED` errors under concurrent sandboxes:
  use the provided launchers. They set `E2B_MAX_KEEPALIVE_CONNECTIONS=512`
  unless the value is already set, which gives the SDK's shared async transport
  a larger keepalive pool for many OpenHands sandboxes.

- Shared memory fills up:
  keep checkpointing disabled or move `AXRL_OUTPUT_DIR` to persistent storage.
