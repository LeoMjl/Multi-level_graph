ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

# Runtime-only dependencies imported by StableToolBench's official fac_eval.py.
RUN python3 -m pip install --no-cache-dir \
    accelerate==0.20.3 \
    pandas==2.2.3

# The upstream image on the A100 host contained a size-correct but corrupted
# NCCL shared object. Reinstall the pinned runtime and fail the build unless
# the resulting library is a real ELF binary.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    nvidia-nccl-cu13==2.29.7 \
    && python3 -c "from pathlib import Path; p = Path('/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2'); assert p.read_bytes()[:4] == b'\\x7fELF', f'invalid NCCL ELF: {p}'"
