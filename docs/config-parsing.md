# Config Parsing Guide

This document explains how AxisRL loads configuration values and how to override them from YAML, command-line arguments, and environment variables.

Relevant implementation:

- Loader: [axrl/utils/config_utils.py](../axrl/utils/config_utils.py)
- Base config models: [axrl/configs.py](../axrl/configs.py)
- Example tests: [tests/utils/test_config_utils.py](../tests/utils/test_config_utils.py)

## Overview

Use `load_and_validate_config(...)` to:

1. load a YAML config file,
2. merge command-line overrides,
3. merge environment variable overrides,
4. validate the final result with Pydantic.

The merge priority is:

$$\text{file} < \text{CLI} < \text{env}$$

That means environment variables win over command-line values, and command-line values win over YAML.

## Basic usage

Typical usage in an entrypoint:

```python
from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
from axrl.utils.config_utils import load_and_validate_config

config = load_and_validate_config(
    GrpoExperimentConfig,
    config_path="path/to/config.yaml",
    print_configs=True,
)
```

If `config_path` is omitted, the loader starts from an empty config and relies on model defaults plus CLI/env overrides.

## 1. YAML file input

You can pass a config file directly:

```python
config = load_and_validate_config(MyConfig, config_path="path/to/train.yaml")
```

Example YAML:

```yaml
a:
  b: 1
flag: false
name: from-file
```

## 2. Command-line overrides

Command-line overrides use dot notation:

```bash
python my_entry.py --a.b=2 --flag=true --name=from-cli
```

This maps to:

```yaml
a:
  b: 2
flag: true
name: from-cli
```

CLI values override the file values.

## 3. Environment variable overrides

Environment variable overrides use the `AXRL__` prefix.

Rules:

- Prefix must be `AXRL__`
- Nested fields use double underscores `__`
- `AXRL__a__b=2` is equivalent to `--a.b=2`

Example:

```bash
export AXRL__a__b=2
export AXRL__flag=true
export AXRL__name=from-env
```

Equivalent CLI form:

```bash
python my_entry.py --a.b=2 --flag=true --name=from-env
```

Since env has highest priority, these values override both YAML and CLI inputs.

## 4. Overriding `config_path`

You can also override the config file path itself.

From CLI:

```bash
python my_entry.py --config_path=path/to/alt.yaml
```

From environment:

```bash
export AXRL__config_path=path/to/prod.yaml
```

Priority still applies:

- explicit `config_path` argument,
- then CLI `--config_path`,
- then env `AXRL__config_path`.

So `AXRL__config_path` has the highest priority.

## 5. Example precedence

End-to-end example with file, CLI, and env together:

Config file `config.yaml`:

```yaml
a:
  b: 1
  c: 2
flag: false
name: from-file
```

Run command:

```bash
export AXRL__a__b=123 && \
python run.py \
  --config_path=config.yaml \
  --a.c=123
```

Key elements:

- `config.yaml` provides the base config
- `--config_path=config.yaml` tells the loader which YAML file to read
- `--a.c=123` overrides `a.c` from the command line
- `AXRL__a__b=123` overrides `a.b` from the environment

Final merged result:

```yaml
a:
  b: 123
  c: 123
flag: false
name: from-file
```


## 6. Type validation

After merging, the config is validated against the provided Pydantic model.

AxisRL config models inherit from `StrictBaseModel` in [axrl/configs.py](../axrl/configs.py), which forbids unknown fields.

This means:

- wrong field names fail fast,
- wrong types fail fast,
- unexpected keys are rejected.

This helps keep experiments reproducible and explicit.

## 7. Notes on values

The loader relies on OmegaConf parsing for CLI and env values.

Common examples:

- `true` / `false` for booleans
- `123` for integers
- `1.5` for floats
- `[1,2,3]` for lists

Examples:

```bash
export AXRL__flag=true
export AXRL__items=[1,2,3]
export AXRL__trainer__learning_rate=1e-5
```

## 8. Related helpers

The same module also provides:

- `save_to_yaml(...)` to serialize a config to YAML
- `config_to_args(...)` to flatten a config into CLI-style arguments

Those helpers are implemented in [axrl/utils/config_utils.py](../axrl/utils/config_utils.py).
