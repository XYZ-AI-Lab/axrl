import asyncio
import logging

from axrl.configs import ModelConfig
from axrl.data.conversation import Conversation
from axrl.data.sample import Sample
from axrl.data.sft_sample_converter import SftSampleConverter
from axrl.example.config_examples import CONV_EXAMPLE_PATH
from axrl.processor.processor_pool import ProcessorPool
from axrl.utils import setup_logger, zst_utils
from axrl.utils.timer import Timer

logger = logging.getLogger(__name__)


def test_create_samples_from_convs() -> None:
    setup_logger("info")

    config = ModelConfig(
        name="Qwen/Qwen3-0.6B",
        seq_length=1024 * 2,
    )

    conversations: list[Conversation] = zst_utils.load_zst(CONV_EXAMPLE_PATH)
    conversations = conversations[:5000]
    for i, conv in enumerate(conversations):
        conv.extra["index"] = i

    speed = create_samples(conversations, config=config)
    speed_batch = batch_create_samples(
        conversations,
        config=config,
        num_proc=8,
    )
    assert speed_batch > speed, f"Expected batch speed {speed_batch} to be greater than single-process speed {speed}"


def create_samples(conversations: list[Conversation], config: ModelConfig) -> float:
    processor = SftSampleConverter(config=config)
    with Timer() as timer:
        samples = [processor.process(conv) for conv in conversations]
        assert len(samples) == len(conversations)
    speed: float = len(samples) / timer.elapsed_seconds
    logger.info(f"Created {len(samples)} samples in {timer.elapsed_seconds:.2f} seconds, {speed:.2f} samples/sec")
    return speed


def batch_create_samples(conversations: list[Conversation], config: ModelConfig, num_proc: int) -> float:
    with (
        Timer() as timer,
        ProcessorPool[Conversation, Sample](
            processor_cls=SftSampleConverter,
            config=config,
            num_processors=num_proc,
        ) as pool,
    ):
        samples = list(asyncio.run(pool.batch_generate(conversations)))
        assert len(samples) == len(conversations)
    speed: float = len(samples) / timer.elapsed_seconds
    logger.info(f"Created {len(samples)} samples in {timer.elapsed_seconds:.2f} seconds, {speed:.2f} samples/sec")
    return speed


if __name__ == "__main__":
    test_create_samples_from_convs()
