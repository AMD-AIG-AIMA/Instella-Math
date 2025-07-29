export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# Install vllm
pip uninstall -y vllm && \
    rm -rf vllm && \
    git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && \
    rm -rf vllm

pip install "tensordict<0.6" --no-deps && \
    pip install accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    numpy \
    pandas \
    peft \
    "pyarrow>=15.0.0" \
    pylatexenc \
    "ray[data,train,tune,serve]" \
    torchdata \
    transformers \
    wandb \
    orjson \
    pybind11
    
pip install antlr4-python3-runtime==4.9.3

cd verl_amd
pip install -e . --no-deps
cd ../

pip install -e .
