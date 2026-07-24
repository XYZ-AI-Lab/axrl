from pydantic import BaseModel

from axrl.utils.logger.metric_logger import REDACTED_CONFIG_VALUE, MetricLogger

FAKE_API_KEY = "not-a-real-secret"


class NestedConfig(BaseModel):
    api_key: str
    max_new_tokens: int


class LoggerConfigForTest(BaseModel):
    nested: NestedConfig
    endpoint: str


def test_flatten_config_redacts_sensitive_keys_without_masking_token_settings() -> None:
    config = LoggerConfigForTest(
        nested=NestedConfig(api_key=FAKE_API_KEY, max_new_tokens=4096),
        endpoint="https://example.invalid",
    )

    flattened = MetricLogger.flatten_config(config)

    assert flattened["nested.api_key"] == REDACTED_CONFIG_VALUE
    assert flattened["nested.max_new_tokens"] == 4096
    assert flattened["endpoint"] == "https://example.invalid"
