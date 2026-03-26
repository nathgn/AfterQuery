FROM nvcr.io/nvidia/pytorch:25.11-py3

# Limit build parallelism to prevent OOM during compilation
ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"

# Clean conflicting pre-installed packages
RUN pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y 2>/dev/null || true

# Install remaining dependencies first (lightweight, caches well)
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Install OpenRLHF (without flash-attn to avoid source compilation)
RUN pip install git+https://github.com/OpenRLHF/OpenRLHF.git --no-build-isolation --no-cache-dir

# Install vllm (brings its own flash-attn prebuilt wheel)
RUN pip install "openrlhf[vllm]" || pip install vllm>=0.13.0

# Copy project into container
COPY . /workspace/afterquery
WORKDIR /workspace/afterquery

# Make scripts executable
RUN chmod +x scripts/*.sh
