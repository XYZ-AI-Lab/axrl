from collections.abc import Callable

from axrl.configs import DatasetConfig
from axrl.datasets.aime2024 import AIME2024
from axrl.datasets.base_dataset import BaseDataset
from axrl.datasets.dapo17k import DAPO17K
from axrl.datasets.flashrag_nq import FlashRAGNQTest, FlashRAGNQTrain
from axrl.datasets.gsm8k import GSM8KTest, GSM8KTrain

DatasetConstructor = Callable[[DatasetConfig | None], BaseDataset]
_DATASET_REGISTRY: dict[str, DatasetConstructor] = {}


def register_dataset(name: str, class_type: DatasetConstructor) -> None:
    """Register a dataset constructor; raises if the name already exists.

    Example:
        register_dataset("custom_math", CustomMathDataset)
        dataset = get_dataset(DatasetConfig(name="custom_math"))
    """
    key = name.lower()
    if key in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    _DATASET_REGISTRY[key] = class_type


def get_dataset(config: DatasetConfig) -> BaseDataset:
    """Return a dataset instance keyed by ``config.name``."""
    key = config.name.lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset name: {config.name}")
    return _DATASET_REGISTRY[key](config)


# Register built-in datasets.
# math
register_dataset("BytedTsinghua-SIA/AIME-2024", AIME2024)
register_dataset("BytedTsinghua-SIA/DAPO-Math-17k", DAPO17K)
register_dataset("openai/gsm8k/train", GSM8KTrain)
register_dataset("openai/gsm8k/test", GSM8KTest)
# search r1
register_dataset("RUC-NLPIR/FlashRAG_datasets/nq/train", FlashRAGNQTrain)
register_dataset("RUC-NLPIR/FlashRAG_datasets/nq/test", FlashRAGNQTest)
