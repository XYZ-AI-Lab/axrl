# Refactoring for the Black Box RL

## Data Flow

![black-box-rl](/docs/figs/black-box-rl.png)

Steps:
- 1. black box harness sends a request to the session router
- 2. session router sends the request to the rollout actor
- 3. rollout actor sends the request to the chat adapter
- 4. chat adapter convert the request to generation input
- 5. rollout actor sends the generation input to the sglang for generation
- 6. sglang generates the generation output
- 7. rollout actor sends the generation output to the chat adapter
- 8. chat adapter converts the generation output to response
- 9. rollout actor sends the response to the session router
- 10. session router sends the response to the black box harness

### Components

Chat Adapter:
- convert the request to generation input
- convert the generation output to response

Session Router:
- route the request to the rollout actor

## Tasks

- [ ] Tested with openhands and codex
- [ ] Tested with two datasets: leetcode and SWE bench
- [ ] Test with response-API.
- [ ] Test with e2b sandbox.
- [ ] Test with GRPO
- [ ] Test with PPO
- [ ] Test with partial rollout to reduce GPU bubbles, with token level and sequence level masking.
- [x] Token in token out
- [x] Prefix merging

## Special cases

The below cases are considered as finished with fail:
- tool call format error
- timeout when waiting request from harness
- verifier timeout
