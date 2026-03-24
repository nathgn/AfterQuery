FROM nvcr.io/nvidia/pytorch:25.11-py3

# Clean conflicting pre-installed packages
RUN pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y 2>/dev/null || true

# Install OpenRLHF from source with vLLM
RUN pip install git+https://github.com/OpenRLHF/OpenRLHF.git
RUN pip install "openrlhf[vllm]" || pip install vllm>=0.13.0

# Install remaining dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Copy project into container
COPY . /workspace/rlvr-terminalbench
WORKDIR /workspace/rlvr-terminalbench

# Make scripts executable
RUN chmod +x scripts/*.sh
