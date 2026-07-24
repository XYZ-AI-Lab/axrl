FROM lmsysorg/sglang:v0.5.14-cu130

RUN pip list --format=freeze

ARG PIP_NO_CACHE_DIR=1

WORKDIR /root/

ENV TZ=Asia/Chongqing
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get purge -y --auto-remove python3-blinker \
    && apt install -y openjdk-21-jdk \
    ripgrep \
    iputils-ping \
    dnsutils \
    curl \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"

RUN pip install pybind11 cmake ninja

RUN pip install nvidia-mathdx==25.6.0

RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

ARG FLASH_ATTENTION_COMMIT=fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout "${FLASH_ATTENTION_COMMIT}" && git submodule update --init && cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf flash-attention/

RUN pip install debugpy

ARG MAGI_ATTENTION_REF=v1.1.1
RUN git clone https://github.com/SandAI-org/MagiAttention.git /tmp/MagiAttention \
    && cd /tmp/MagiAttention \
    && git checkout "${MAGI_ATTENTION_REF}" \
    && git submodule update --init --recursive \
    && MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90,100 pip install --no-build-isolation . \
    && rm -rf /tmp/MagiAttention

RUN pip install flash-linear-attention==0.4.1

RUN pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation

RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

ARG APEX_COMMIT=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
RUN NVCC_APPEND_FLAGS="--threads 4" \
    MAX_JOBS=128 pip install -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}

ARG TRANSFORMER_ENGINE_REF=release_v2.10
RUN export NVTE_FRAMEWORK=pytorch && \
    MAX_JOBS=128 NVTE_BUILD_THREADS_PER_JOB=4 \
    pip3 install --resume-retries 999 \
    --no-cache-dir \
    --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_REF}


RUN pip install 'nvidia-modelopt[torch]>=0.44' --no-build-isolation

ARG FAISS_VERSION=v1.14.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends swig libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch "${FAISS_VERSION}" https://github.com/facebookresearch/faiss.git /tmp/faiss \
    && cmake -B /tmp/faiss/build /tmp/faiss \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DPython_EXECUTABLE="$(command -v python3)" \
    && make -C /tmp/faiss/build -j"$(nproc)" faiss swigfaiss \
    && make -C /tmp/faiss/build install \
    && install -m 755 /tmp/faiss/build/faiss/python/libfaiss_python_callbacks.so /usr/local/lib/libfaiss_python_callbacks.so \
    && ldconfig \
    && cd /tmp/faiss/build/faiss/python \
    && python3 setup.py install \
    && python3 -c "import faiss; print(hasattr(faiss, 'IndexFlatL2'), hasattr(faiss, 'GpuMultipleClonerOptions'))" \
    && rm -rf /tmp/faiss

RUN pip install \
    accelerate \
    bs4 \
    chromadb \
    colorama \
    dacite \
    datasets \
    jsonlines \
    json-repair \
    langchain \
    langchain_community \
    langchain_chroma \
    langchain_huggingface \
    markdown \
    markdownify \
    math-verify \
    matplotlib \
    mypy \
    ollama \
    omegaconf \
    openpyxl \
    pandas \
    pandas-stubs \
    pickledb \
    Pillow \
    plotly \
    pre-commit \
    pydantic \
    pylatexenc \
    pymupdf \
    "pyright[nodejs]" \
    pyserini \
    pytest \
    python-docx \
    qwen_vl_utils \
    packaging \
    rank_bm25 \
    "ray[default]>=2.54.0" \
    rich \
    rouge \
    ruff \
    scikit-learn \
    seaborn \
    sentence-transformers \
    spacy \
    sqlmodel \
    streamlit \
    tensorboard \
    tensordict \
    torchdata \
    tqdm \
    tqdm-stubs \
    TransferQueue \
    "transformers==5.8.1" \
    types-protobuf \
    types-PyYAML \
    types-requests \
    unstructured \
    wandb \
    wandb-workspaces \
    word2number \
    zstandard

ARG MEGATRON_REF=core_v0.18.0
RUN pip install --no-build-isolation --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_REF}

ARG MEGATRON_BRIDGE_REF=v0.5.0
RUN pip install --no-deps --no-cache-dir git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@${MEGATRON_BRIDGE_REF}

ARG TORCH_MEMORY_SAVER_COMMIT=a193d9dd1b877d33c64a41cfb3db9f867df2d926
RUN git clone https://github.com/fzyzcjy/torch_memory_saver.git /tmp/torch_memory_saver \
    && cd /tmp/torch_memory_saver \
    && git checkout "${TORCH_MEMORY_SAVER_COMMIT}" \
    && TMS_CUDA_MAJOR=13 pip install --no-cache-dir . \
    && rm -rf /tmp/torch_memory_saver

RUN pip install --no-cache-dir --upgrade nvidia-cudnn-cu13==9.19.0.56

RUN ( \
    cd && \
    git clone https://github.com/gpakosz/.tmux.git && \
    ln -s -f .tmux/.tmux.conf && \
    cp .tmux/.tmux.conf.local . \
    )

RUN pip install --no-cache-dir --upgrade "transformers==5.8.1"

RUN set -eux; \
    printf '%s\n' \
        'import importlib.util' \
        'import inspect' \
        '' \
        'import megatron.bridge.models.qwen.qwen35_bridge as qwen35_bridge' \
        'import megatron.core._rank_utils as rank_utils' \
        'import megatron.core.ssm.gated_delta_net as gdn' \
        '' \
        'assert importlib.util.find_spec("fla") is not None, "flash-linear-attention is not importable"' \
        'assert getattr(gdn, "HAVE_FLA", False), "Megatron GDN did not detect FLA"' \
        'assert hasattr(rank_utils, "safe_get_world_size"), "Megatron Core lacks safe_get_world_size required by Bridge"' \
        '' \
        'gdn_forward = inspect.getsource(gdn.GatedDeltaNet.forward)' \
        'assert "packed_seq_params" in gdn_forward, "Megatron GDN lacks packed_seq_params support"' \
        'assert "qkv_format == " + repr("thd") in gdn_forward, "Megatron GDN lacks THD packed path"' \
        '' \
        'bridge_source = inspect.getsource(qwen35_bridge)' \
        'required_bridge_strings = (' \
        '    "qwen3_5_moe_text",' \
        '    "qwen3_5_text",' \
        '    "mtp_experts_packed",' \
        '    "GDNLinearMappingSeparate",' \
        '    "get_transformer_block_with_experimental_attention_variant_spec",' \
        ')' \
        'missing = [needle for needle in required_bridge_strings if needle not in bridge_source]' \
        'assert not missing, "Qwen3.5/3.6 text bridge missing " + ", ".join(missing)' \
        '' \
        'print("Qwen3.6 text Bridge/MCore smoke check passed")' \
        > /tmp/qwen36_bridge_smoke.py; \
    python /tmp/qwen36_bridge_smoke.py; \
    rm /tmp/qwen36_bridge_smoke.py


RUN set -eux; \
    DEB=libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb; \
    curl -fSL "https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2013.0/ubuntu24_04/x64/${DEB}" -o "/tmp/${DEB}"; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y "/tmp/${DEB}"; \
    rm -f "/tmp/${DEB}"; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL https://install.openhands.dev/install.sh | sh

RUN pip list --format=freeze
