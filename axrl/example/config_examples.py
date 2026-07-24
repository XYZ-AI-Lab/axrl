from axrl.configs import AXRL_DIR, MCoreOptimizerConfig, MegatronWorkerConfig, ModelConfig

CONV_EXAMPLE_PATH = AXRL_DIR.data / "processed_data" / "conv-example.zst"


def get_megatron_trainer_config(
    tp_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    vpp_size: int | None = None,
    model_config: ModelConfig | None = None,
    *,
    fp16: bool = False,
    bf16: bool = True,
) -> MegatronWorkerConfig:
    """Get a sample MegatronWorkerConfig for training."""
    if model_config is None:
        model_config = ModelConfig(
            name="Qwen/Qwen3-0.6B",
            seq_length=1024 * 2,
        )
    megatraon_trainer_config = MegatronWorkerConfig(
        model=model_config,
        num_epochs=20,
        train_micro_batch_size=2,
        global_batch_size=32,
        tp_size=tp_size,
        dp_size=dp_size,
        pp_size=pp_size,
        vpp_size=vpp_size,
        cp_size=cp_size,
        fp16=fp16,
        bf16=bf16,
        use_gloo_process_groups=True,
        optimizer=MCoreOptimizerConfig(
            lr=1e-4,
            min_lr=1e-6,
            bf16=bf16,
        ),
    )
    return megatraon_trainer_config
