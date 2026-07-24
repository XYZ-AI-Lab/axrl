# AXRL OpenHands E2B Template

This template is intentionally small: Python 3.12 slim, basic shell tools, and
the OpenHands CLI installed by the upstream installer.

Build it from this directory:

```bash
cd axis_recipe/blackbox_rl/e2b_template
e2b template build --name axrl-openhands
```

The AXRL repo, model weights, and datasets are not copied into the sandbox.
OpenHands only writes and runs the generated LeetCode solution in `/workspace`
and calls the AXRL OpenAI-compatible proxy through the configured exposed URL.
