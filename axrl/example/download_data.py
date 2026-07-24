from axrl.configs import HfDataConfig, ModelConfig
from axrl.utils.hf.download_data_from_hf import download_data_file
from axrl.utils.hf.download_model_from_hf import download_model


def download_all() -> None:
    datas: list[HfDataConfig] = [
        HfDataConfig(
            repo_id="open-r1/DAPO-Math-17k-Processed",
            filename="all/train-00000-of-00001.parquet",
        ),
        HfDataConfig(
            repo_id="bespokelabs/Bespoke-Stratos-17k",
            filename="data/train-00000-of-00001.parquet",
        ),
        HfDataConfig(
            repo_id="BytedTsinghua-SIA/AIME-2024",
            filename="data/aime-2024.parquet",
        ),
        HfDataConfig(
            repo_id="BytedTsinghua-SIA/DAPO-Math-17k",
            filename="data/dapo-math-17k.parquet",
        ),
        HfDataConfig(
            repo_id="RUC-NLPIR/FlashRAG_datasets",
            filename="nq/train.jsonl",
        ),
        HfDataConfig(
            repo_id="RUC-NLPIR/FlashRAG_datasets",
            filename="nq/test.jsonl",
        ),
        HfDataConfig(
            repo_id="openai/gsm8k",
            filename="main/train-00000-of-00001.parquet",
        ),
        HfDataConfig(
            repo_id="openai/gsm8k",
            filename="main/test-00000-of-00001.parquet",
        ),
        HfDataConfig(
            repo_id="newfacade/LeetCodeDataset",
            filename="LeetCodeDataset-train.jsonl",
        ),
        HfDataConfig(
            repo_id="newfacade/LeetCodeDataset",
            filename="LeetCodeDataset-test.jsonl",
        ),
    ]

    for data in datas:
        download_data_file(config=data)

    models: list[ModelConfig] = [
        ModelConfig(name="Qwen/Qwen3-0.6B-Base"),
        ModelConfig(name="Qwen/Qwen3-0.6B"),
        ModelConfig(name="Qwen/Qwen3-1.7B"),
        ModelConfig(name="Qwen/Qwen3-8B"),
        ModelConfig(name="Qwen/Qwen3-4B-Instruct-2507"),
        ModelConfig(name="Qwen/Qwen2.5-1.5B-Instruct"),
        ModelConfig(name="Qwen/Qwen2.5-3B-Instruct"),
        ModelConfig(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        ModelConfig(name="Qwen/Qwen3-30B-A3B-Instruct-2507"),
        ModelConfig(name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"),
        ModelConfig(name="Qwen/Qwen3-30B-A3B-Base"),
    ]
    for model in models:
        download_model(config=model)


if __name__ == "__main__":
    download_all()
