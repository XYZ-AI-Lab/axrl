#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

pre-commit install
pre-commit run --all-files --verbose

# echo "pytest.."
# pytest -v

echo "pyright checking.."
pyright .

echo "mypy checking.."
mypy --config-file pyproject.toml ./ --no-incremental

echo "All checks passed successfully!"
