#!/usr/bin/bash

set -Eeuo pipefail

log_test_harness_event() {
	printf '%s [run-tests] %s\n' "$(date --iso-8601=seconds)" "$*"
}

on_exit() {
	local exit_code=$?
	log_test_harness_event "script exiting with code ${exit_code}"
}

on_sigterm() {
	log_test_harness_event "received SIGTERM"
	exit 143
}

on_sigint() {
	log_test_harness_event "received SIGINT"
	exit 130
}

trap on_exit EXIT
trap on_sigterm TERM
trap on_sigint INT

LOCK_FILE=/tmp/axrl-run-tests.lock
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
	echo "Another test run is already active on this node. Lock: ${LOCK_FILE}" >&2
	exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
export AXRL_OUTPUT_DIR="${AXRL_OUTPUT_DIR:-${HOME}/axrl-data/outputs/default}"
OUTPUT_DIR="${AXRL_OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run-tests-${TIMESTAMP}.log"
PYTEST_INFO_LOG_FILE="${AXRL_PYTEST_LOG_FILE:-${OUTPUT_DIR}/pytest-info-${TIMESTAMP}.log}"

mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log_test_harness_event "writing detailed test log to ${LOG_FILE}"
log_test_harness_event "writing pytest INFO log to ${PYTEST_INFO_LOG_FILE}"

python -u -m axrl.example.download_data
python -u -m axrl.example.prepare_conv_data

# Use --lf (last failed) to only re-run previously failed tests.
# On first run (no cache), all tests run normally.
# -x: stop after first failure, -vv: show detailed traceback
# shellcheck disable=SC2102
# pytest tests -vv -x --lf 2>&1 | tee "${OUTPUT_DIR}/run-tests-${TIMESTAMP}.log"
log_test_harness_event "starting pytest with unbuffered output, stdout passthrough, and faulthandler enabled"
python -u -X faulthandler -m pytest tests -vv -s \
	--log-level=INFO \
	--log-cli-level=INFO \
	--log-cli-format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s" \
	--log-file="${PYTEST_INFO_LOG_FILE}" \
	--log-file-level=INFO \
	--log-file-format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s"
# to see full log
# pytest tests -vv -x -s 2>&1 | tee tmp/tests.log
